"""
Build a NEW query workload for the MoRe_large dataset
(== imdb_data/final/10000/MoRe, downloaded/converted into
 data/datasets/MoRe_large with columns train_mvector/train_avgrating/train_mid
 and train_rvector/train_total_votes/train_rid).

What it produces (under data/datasets/MoRe_large/):
  queries_new/queries_flex_movies_sim_0_<fid>.hdf5   keys: test, mid,  mids,  distances
  queries_new/queries_flex_reviews_sim_0_<fid>.hdf5  keys: test, rid,  rids,  distances
  filters_new/movies_filters_0.hdf5                  keys: filters, selectivities
  filters_new/reviews_filters_0.hdf5                 keys: filters, selectivities
  stats_new/filter_stats_0_k2048.csv                 cols: q_id,query_type,filter,
                                                            selectivity,correlation_ACORN,correlation_NEW

Workload spec
  * 100 query vectors per dataset, sampled (seed=42) from existing dataset points.
  * Filters on avg_rating (movies) and total_votes (reviews) at percentile-based
    target selectivities, plus a No_filter baseline. Filters whose percentile
    threshold collides with another are de-duplicated.
  * "True results": exact top-k (k=100) neighbours WITHIN each filtered subset
    (cosine similarity == inner product on L2-normalised vectors), computed on GPU.
  * GLS correlation (correlation_NEW) computed with the k=2048 neighbourhood method
    (local-vs-global selectivity ratio mapped to [-1, 1]); correlation_ACORN
    (min-distance difference vs a random equal-size sample) is also recorded so the
    CSV stays compatible with dataset_corr_stats.py.

Runs on GPU via faiss-gpu (torch CUDA is unavailable on this box due to a driver/cu130
mismatch, but faiss-gpu sees all 8 Tesla P40s).
"""

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import faiss


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = "data/datasets/MoRe_large"
MODE = 0
SEED = 42
N_QUERIES = 100          # query vectors sampled per dataset
K_TRUE = 100             # k for the true-neighbour results stored per query
K_GLS = 2048             # neighbourhood size for GLS / correlation calculation
EPS = 1e-6

# "0.75" in the request is read as "0.075" (it sits between 0.05 and 0.1).
TARGET_SELECTIVITIES = [
    0.005, 0.01, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25,
    0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9,
]

STR_DT = h5py.string_dtype(encoding="utf-8")

# one StandardGpuResources per device so big indexes can live on different GPUs
_GPU_RES: dict[int, "faiss.StandardGpuResources"] = {}


def gpu_res(device: int) -> "faiss.StandardGpuResources":
    if device not in _GPU_RES:
        r = faiss.StandardGpuResources()
        _GPU_RES[device] = r
    return _GPU_RES[device]


def build_gpu_flat_ip(vecs: np.ndarray, device: int) -> "faiss.GpuIndexFlatIP":
    """Build an exact inner-product (== cosine for unit vectors) index on `device`."""
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = device
    index = faiss.GpuIndexFlatIP(gpu_res(device), vecs.shape[1], cfg)
    index.add(np.ascontiguousarray(vecs, dtype=np.float32))
    return index


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_dataset(kind: str):
    """kind in {'movies','reviews'} -> (ids[list[str]], vecs[float64], filt_col[np.ndarray], col_name)."""
    if kind == "movies":
        path = f"{DATA_DIR}/datasets/movies_dataset_{MODE}.hdf5"
        with h5py.File(path, "r") as f:
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["train_mid"][:]]
            vecs = f["train_mvector"][:]
            filt_col = f["train_avgrating"][:].astype(np.float64)
        return ids, vecs, filt_col, "avg_rating"
    else:
        path = f"{DATA_DIR}/datasets/reviews_dataset_{MODE}.hdf5"
        with h5py.File(path, "r") as f:
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["train_rid"][:]]
            vecs = f["train_rvector"][:]
            filt_col = f["train_total_votes"][:].astype(np.float64)
        return ids, vecs, filt_col, "total_votes"


# ----------------------------------------------------------------------------
# Filter construction (percentile based, de-duplicated by threshold)
# ----------------------------------------------------------------------------
def build_filters(filt_col: np.ndarray, col_name: str):
    """Return list of (filter_string, threshold_or_None, selectivity, filtered_idx)."""
    n = len(filt_col)
    full_idx = np.arange(n)
    out = [("No_filter", None, 1.0, full_idx)]

    seen_thr = set()
    for t in TARGET_SELECTIVITIES:
        thr = float(np.nanpercentile(filt_col, 100.0 - t * 100.0))
        if thr in seen_thr:
            continue
        seen_thr.add(thr)
        idx = np.nonzero(filt_col >= thr)[0]
        sel = len(idx) / n
        out.append((f"{col_name} >= {thr}", thr, sel, idx))
    return out


def save_filters(filters, out_path: str):
    fstrings = [f[0] for f in filters]
    sels = [f[2] for f in filters]
    with h5py.File(out_path, "w") as f:
        f.create_dataset("filters", data=np.array(fstrings, dtype=STR_DT))
        f.create_dataset("selectivities", data=np.array(sels, dtype=np.float64))


# ----------------------------------------------------------------------------
# True neighbours (top-K_TRUE within filtered subset)
# ----------------------------------------------------------------------------
def save_queries(kind, fid, query_vecs, query_ids, neigh_ids, neigh_dists, out_dir):
    id_key = "mid" if kind == "movies" else "rid"
    ids_key = "mids" if kind == "movies" else "rids"
    name = "movies" if kind == "movies" else "reviews"
    path = f"{out_dir}/queries_flex_{name}_sim_{MODE}_{fid}.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("test", data=np.asarray(query_vecs, dtype=np.float64))
        f.create_dataset(id_key, data=np.array(query_ids, dtype=STR_DT))
        f.create_dataset(ids_key, data=np.array(neigh_ids, dtype=STR_DT))
        f.create_dataset("distances", data=np.asarray(neigh_dists, dtype=np.float64))


# ----------------------------------------------------------------------------
# GLS / ACORN correlation for one filter, reusing precomputed full-dataset radii
# ----------------------------------------------------------------------------
def correlations_for_filter(s_filt, max_dist_full, s_samp_top1, selectivity, max_k):
    """Return (acorn_list, gls_list) for all query vectors of one filter.

    ACORN: (top-1 sim on a random equal-size sample) - (top-1 sim on filtered set).
    GLS  : local selectivity (fraction of filtered top-k inside the full top-k radius)
           vs global selectivity, mapped to [-1, 1] via (r-1)/(r+1).

    s_filt        : (N, max_k) filtered-subset similarities (descending).
    max_dist_full : (N,)       per-query radius (1 - min full-top-k sim), filter independent.
    s_samp_top1   : (N,)       top-1 sim of a random equal-size sample (for ACORN).
    """
    s_filt1 = s_filt[:, 0]
    acorn = (s_samp_top1 - s_filt1).astype(np.float64)

    # count filtered top-k points lying OUTSIDE the full-dataset top-k radius
    outside = (1.0 - s_filt) > (max_dist_full[:, None] + EPS)
    count_outside = outside.sum(axis=1)
    sel_around = (max_k - count_outside) / max_k

    gls = np.zeros(len(s_filt), dtype=np.float64)
    if selectivity > 0:
        ratio = sel_around / selectivity
        gls = (ratio - 1.0) / (ratio + 1.0)
    return acorn.tolist(), gls.tolist()


# ----------------------------------------------------------------------------
# Per-dataset driver
# ----------------------------------------------------------------------------
def process(kind, q_type, out_queries, full_dev, filt_dev, samp_dev, stats_rows):
    print(f"\n========== {kind} ({q_type}) ==========", flush=True)
    ids, vecs, filt_col, col_name = load_dataset(kind)
    ids = np.array(ids, dtype=object)
    n, d = vecs.shape
    print(f"  rows={n}  dim={d}  filter_col={col_name}", flush=True)

    rng = np.random.RandomState(SEED)
    q_idx = rng.choice(n, N_QUERIES, replace=False)
    q_idx.sort()
    query_vecs = vecs[q_idx]
    query_ids = ids[q_idx].tolist()
    q32 = np.ascontiguousarray(query_vecs, dtype=np.float32)

    filters = build_filters(filt_col, col_name)
    print(f"  {len(filters)} filters (incl. No_filter):", flush=True)
    for fstr, _thr, sel, idx in filters:
        print(f"    {fstr:<28} sel={sel:.5f}  (|F|={len(idx)})", flush=True)

    vecs32 = np.ascontiguousarray(vecs, dtype=np.float32)

    # full-dataset top-K_GLS radius per query: filter independent, computed ONCE
    full_index = build_gpu_flat_ip(vecs32, full_dev)
    s_full, _ = full_index.search(q32, min(K_GLS, n))
    max_dist_full = 1.0 - np.min(s_full, axis=1)
    del full_index, s_full

    for fid, (fstr, _thr, sel, fidx) in enumerate(filters):
        t0 = datetime.now()
        filt_index = build_gpu_flat_ip(vecs32[fidx], filt_dev)

        # one wide search reused for both true results (top-K_TRUE) and GLS (top-K_GLS)
        k_search = min(K_GLS, len(fidx))
        sims, nn = filt_index.search(q32, k_search)

        # --- true top-K_TRUE neighbours within the filtered subset ---
        k_true = min(K_TRUE, len(fidx))
        neigh_ids = ids[fidx][nn[:, :k_true]]            # (N_QUERIES, k_true) object array
        save_queries(kind, fid, query_vecs, query_ids,
                     neigh_ids, sims[:, :k_true].astype(np.float64), out_queries)

        # --- ACORN: random equal-size sample, top-1 similarity ---
        samp = rng.choice(n, len(fidx), replace=False)
        sample_index = build_gpu_flat_ip(vecs32[samp], samp_dev)
        s_samp, _ = sample_index.search(q32, 1)
        del sample_index

        acorn, gls = correlations_for_filter(sims, max_dist_full, s_samp[:, 0],
                                             sel, k_search)
        for pos in range(N_QUERIES):
            stats_rows.append({
                "q_id": pos,
                "query_type": q_type,
                "filter": fstr,
                "selectivity": sel,
                "correlation_ACORN": acorn[pos],
                "correlation_NEW": gls[pos],
            })

        del filt_index
        dt = (datetime.now() - t0).total_seconds()
        print(f"  [fid {fid:>2}] {fstr:<28} sel={sel:.4f}  k_true={k_true} k_gls={k_search}"
              f"  done in {dt:.1f}s", flush=True)

    return filters


# ----------------------------------------------------------------------------
def main():
    global DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    start = datetime.now()
    print("Started:", start.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"faiss sees {faiss.get_num_gpus()} GPUs", flush=True)

    out_queries = f"{DATA_DIR}/queries_new"
    out_filters = f"{DATA_DIR}/filters_new"
    out_stats = f"{DATA_DIR}/stats_new"
    for p in (out_queries, out_filters, out_stats):
        os.makedirs(p, exist_ok=True)

    stats_rows = []

    movie_filters = process("movies", "flex_movies_sim",
                            out_queries, full_dev=0, filt_dev=1, samp_dev=2,
                            stats_rows=stats_rows)
    save_filters(movie_filters, f"{out_filters}/movies_filters_{MODE}.hdf5")

    review_filters = process("reviews", "flex_reviews_sim",
                             out_queries, full_dev=3, filt_dev=4, samp_dev=5,
                             stats_rows=stats_rows)
    save_filters(review_filters, f"{out_filters}/reviews_filters_{MODE}.hdf5")

    stats_df = pd.DataFrame(stats_rows,
                            columns=["q_id", "query_type", "filter", "selectivity",
                                     "correlation_ACORN", "correlation_NEW"])
    stats_path = f"{out_stats}/filter_stats_{MODE}_k{K_GLS}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved stats -> {stats_path}  ({len(stats_df)} rows)")

    n_movie_combos = N_QUERIES * len(movie_filters)
    n_review_combos = N_QUERIES * len(review_filters)
    print(f"\nWorkload summary")
    print(f"  movies : {len(movie_filters)} filters x {N_QUERIES} queries = {n_movie_combos} combos")
    print(f"  reviews: {len(review_filters)} filters x {N_QUERIES} queries = {n_review_combos} combos")
    print(f"  total vec-filter combos = {n_movie_combos + n_review_combos}")

    end = datetime.now()
    print("Finished:", end.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed (s):", (end - start).total_seconds())


if __name__ == "__main__":
    main()
