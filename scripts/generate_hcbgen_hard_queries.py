#!/usr/bin/env python3
"""Generate hardness-controlled MoReVec queries via HCBGen Algorithm 2 (Match-PDF).

Adapts Fine-Grained Hardness-Controlled Query Generation (Match-PDF) with
Budgeted Closest-Fill Fallback to MoReVec's real metadata filters:

  * Sample qf from the existing MoRe filter set (range predicates on
    avg_rating / total_votes) -- we do NOT synthesize categorical labels.
  * Sample qv from the base dataset (self excluded from k-NN estimates).
  * Accept/reject into hardness bins; leftover slots filled by closest-fill.

Target hardness PDF: Uniform on [0, 10] with 10 bins x 100 queries (=1000).

Hardness metric is HCBGen's published Post_Hardness (calculate_hardness_v5_1.py),
NOT the intermediate H_scan/alpha alone:

  selectivity_term = log10(N / |F|) + 1
  density          = d(q, K-NN | F) / d(q, K-NN | V0)     (K=10)
  H_scan (=alpha)  = selectivity_term * density * 10
  H_fetch          = 1 + latency_ms(HNSW search depth=int(H_scan))
  Post_Hardness    = H_scan * H_fetch / 100          <-- Match-PDF control signal

Paper semi-real hardness histograms (~0-1.5) are Post_Hardness. H_scan alone
is typically ~10-40 before the /100 scaling.

GLS correlation (arXiv:2602.11443) is stored alongside for every accepted query.

Output (under data/datasets/MoRe_large/hard_queries/):
  {table}_hcbgen_match_pdf.hdf5
  {table}_hcbgen_match_pdf_meta.csv
  {table}_gls_vs_post_{tag}.png
  target_pdf_{tag}.json

Prefer faiss-gpu (fanns_gpu env) for exact k-NN + HNSW H_fetch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    import faiss
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "faiss is required. Use conda env `fanns_gpu` (faiss-gpu) or "
        "`ann-hq` with faiss-cpu installed."
    ) from e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "datasets" / "MoRe_large"
STR_DT = h5py.string_dtype(encoding="utf-8")
K_ALPHA = 10
K_GLS = 2048
K_TRUE = 100
EPS = 1e-12

_PRED_RE = re.compile(r"^\s*(.+?)\s*(>=|<=|==|=|>|<)\s*(.+?)\s*$")


def has_gpu() -> bool:
    try:
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


_GPU_RES: dict[int, "faiss.StandardGpuResources"] = {}


def gpu_res(device: int = 0):
    if device not in _GPU_RES:
        _GPU_RES[device] = faiss.StandardGpuResources()
    return _GPU_RES[device]


def build_flat_l2(vecs: np.ndarray, device: int | None):
    """Exact L2 index on GPU if available, else CPU."""
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    d = vecs.shape[1]
    if device is not None and has_gpu():
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = device
        index = faiss.GpuIndexFlatL2(gpu_res(device), d, cfg)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(vecs)
    return index


def build_hnsw(vecs: np.ndarray, M: int = 32, ef_c: int = 50, ef_s: int = 50):
    d = vecs.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_c
    bs = 50_000
    for i in range(0, len(vecs), bs):
        index.add(np.ascontiguousarray(vecs[i : i + bs], dtype=np.float32))
        print(f"    HNSW added {min(i + bs, len(vecs)):,}/{len(vecs):,}", flush=True)
    index.hnsw.efSearch = ef_s
    return index


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_table(data_dir: Path, kind: str):
    if kind == "movies":
        path = data_dir / "datasets" / "movies_dataset_0.hdf5"
        with h5py.File(path, "r") as f:
            ids = np.array(
                [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["train_mid"][:]],
                dtype=object,
            )
            vecs = np.ascontiguousarray(f["train_mvector"][:], dtype=np.float32)
            col = f["train_avgrating"][:].astype(np.float64)
        return ids, vecs, col, "avg_rating", "flex_movies_sim"
    path = data_dir / "datasets" / "reviews_dataset_0.hdf5"
    with h5py.File(path, "r") as f:
        ids = np.array(
            [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["train_rid"][:]],
            dtype=object,
        )
        vecs = np.ascontiguousarray(f["train_rvector"][:], dtype=np.float32)
        col = f["train_total_votes"][:].astype(np.float64)
    return ids, vecs, col, "total_votes", "flex_reviews_sim"


def load_filters(data_dir: Path, kind: str):
    path = data_dir / "filters" / f"{kind}_filters_0.hdf5"
    out = []
    with h5py.File(path, "r") as f:
        for s, flt in zip(f["selectivities"][:], f["filters"][:]):
            flt = flt.decode() if isinstance(flt, (bytes, np.bytes_)) else str(flt)
            if flt.lower() in {"no_filter", "nofilter"}:
                continue
            out.append((flt, float(s)))
    return out


def parse_threshold(filter_str: str) -> float:
    m = _PRED_RE.match(filter_str)
    if not m:
        raise ValueError(f"Cannot parse filter: {filter_str}")
    return float(m.group(3))


def filtered_indices(col: np.ndarray, filter_str: str) -> np.ndarray:
    thr = parse_threshold(filter_str)
    # MoRe filters are all "attr >= thr"
    return np.nonzero(col >= thr)[0]


# ---------------------------------------------------------------------------
# Hardness + GLS
# ---------------------------------------------------------------------------
def knn_kth_excluding_self(
    index,
    queries: np.ndarray,
    q_global_ids: np.ndarray,
    k: int,
    id_map: np.ndarray | None = None,
) -> np.ndarray:
    """Return L2 distance to the k-th neighbour, excluding the query itself.

    id_map: if the index was built on a subset, maps local index -> global id.
    """
    nq = len(queries)
    D, I = index.search(np.ascontiguousarray(queries, dtype=np.float32), k + 1)
    out = np.full(nq, np.nan, dtype=np.float64)
    for i in range(nq):
        dists = []
        for j in range(I.shape[1]):
            loc = int(I[i, j])
            if loc < 0:
                continue
            gid = int(id_map[loc]) if id_map is not None else loc
            if gid == int(q_global_ids[i]):
                continue
            dists.append(float(np.sqrt(max(D[i, j], 0.0))))
            if len(dists) == k:
                break
        if len(dists) == k:
            out[i] = dists[-1]
    return out


def compute_alpha_batch(
    d0: np.ndarray,
    df: np.ndarray,
    n_total: int,
    n_filt: int,
) -> np.ndarray:
    """HCBGen H_scan / alpha. Matches calculate_hardness_v5_1.compute_H_scan."""
    sel_term = np.log10(n_total / max(n_filt, 1)) + 1.0
    # mirror reference: if d0 == 0, replace with df (density -> 1)
    d0_safe = d0.copy()
    zero = d0_safe <= EPS
    d0_safe[zero] = df[zero]
    density = df / (d0_safe + 1e-9)
    alpha = sel_term * density * float(K_ALPHA)
    alpha[~np.isfinite(alpha)] = 10.0
    return alpha.astype(np.float64)


def compute_h_fetch_batch(hnsw, queries: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """H_fetch = 1 + latency_ms of HNSW search with depth k=int(alpha)."""
    out = np.zeros(len(queries), dtype=np.float64)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    for i in range(len(q)):
        k = int(np.clip(alphas[i], 1, 5000))
        t0 = time.perf_counter()
        hnsw.search(q[i : i + 1], k)
        out[i] = 1.0 + (time.perf_counter() - t0) * 1000.0
    return out


def gls_from_full_neighbors(
    full_neighbor_ids: np.ndarray,
    filter_mask: np.ndarray,
    selectivity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact GLS (arXiv:2602.11443): rho = (sigma_l - sigma_g)/(sigma_l + sigma_g).

    full_neighbor_ids : (nq, K) global ids of the unfiltered k-NN (self excluded);
                        unused slots are -1 and ignored in the mean.
    filter_mask       : (N,) bool, True iff base vector passes the filter
    sigma_l           = fraction of those neighbours that pass the filter
    """
    ids = full_neighbor_ids.copy()
    valid = ids >= 0
    ids[~valid] = 0
    passed = filter_mask[ids]
    passed[~valid] = False
    counts = valid.sum(axis=1).clip(min=1)
    sigma_l = passed.sum(axis=1) / counts
    if selectivity <= 0:
        rho = np.zeros(len(sigma_l), dtype=np.float64)
    else:
        rho = (sigma_l - selectivity) / (sigma_l + selectivity + 1e-15)
    return rho.astype(np.float64), sigma_l.astype(np.float64)


# ---------------------------------------------------------------------------
# Algorithm 2: Match-PDF with budgeted closest-fill
# ---------------------------------------------------------------------------
def match_pdf(
    hardness: np.ndarray,
    bin_edges: np.ndarray,
    target_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slot-driven Match-PDF over a precomputed candidate pool (HCBGen Alg. 2).

    Pass 1: accept candidates whose hardness falls in a bin with remaining slots.
    Pass 2 (closest-fill): for each unused candidate (hardest first), assign it to
    the nearest remaining bin -- matching HCBGen's per-query nearest-available-bin
    fallback so hard queries preferentially fill high bins.
    """
    num_bins = len(target_counts)
    remaining = target_counts.astype(int).copy()
    n = len(hardness)

    def bin_of(h: float) -> int:
        b = int(np.digitize([h], bin_edges, right=False)[0]) - 1
        return int(np.clip(b, 0, num_bins - 1))

    def dist_to_interval(x: float, lo: float, hi: float) -> float:
        if lo <= x <= hi:
            return 0.0
        return (lo - x) if x < lo else (x - hi)

    def nearest_available_bin(x: float) -> tuple[int, float]:
        best_b, best_d = -1, float("inf")
        for b in range(num_bins):
            if remaining[b] <= 0:
                continue
            d = dist_to_interval(x, float(bin_edges[b]), float(bin_edges[b + 1]))
            if d < best_d or (d == best_d and remaining[b] > remaining[best_b]):
                best_b, best_d = b, d
        return best_b, best_d

    selected: list[int] = []
    assigned_bins: list[int] = []
    fill_method: list[int] = []
    used = np.zeros(n, dtype=bool)

    # Pass 1: exact bin hits (random order to avoid filter bias)
    order = np.random.permutation(n)
    leftovers: list[int] = []
    for idx in order:
        if int(remaining.sum()) <= 0:
            break
        h = float(hardness[idx])
        if not np.isfinite(h):
            continue
        b = bin_of(h)
        if remaining[b] > 0:
            remaining[b] -= 1
            used[idx] = True
            selected.append(idx)
            assigned_bins.append(b)
            fill_method.append(0)
        else:
            leftovers.append(idx)

    # Pass 2: closest-fill -- assign unused candidates (hardest first) to nearest
    # remaining bin so high-hardness queries fill high empty bins.
    leftovers = [i for i in leftovers if not used[i]]
    leftovers.sort(key=lambda i: float(hardness[i]), reverse=True)
    for idx in leftovers:
        if int(remaining.sum()) <= 0:
            break
        h = float(hardness[idx])
        b, _ = nearest_available_bin(h)
        if b < 0:
            break
        remaining[b] -= 1
        used[idx] = True
        selected.append(idx)
        assigned_bins.append(b)
        fill_method.append(1)

    # Pass 3: if still short, take any unused finite candidates
    if int(remaining.sum()) > 0:
        unused = [i for i in range(n) if (not used[i]) and np.isfinite(hardness[i])]
        unused.sort(key=lambda i: float(hardness[i]), reverse=True)
        for idx in unused:
            if int(remaining.sum()) <= 0:
                break
            h = float(hardness[idx])
            b, _ = nearest_available_bin(h)
            if b < 0:
                break
            remaining[b] -= 1
            used[idx] = True
            selected.append(idx)
            assigned_bins.append(b)
            fill_method.append(1)

    return (
        np.asarray(selected, dtype=int),
        np.asarray(assigned_bins, dtype=int),
        np.asarray(fill_method, dtype=int),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def process_table(
    kind: str,
    data_dir: Path,
    out_dir: Path,
    n_queries: int,
    n_bins: int,
    h_min: float,
    h_max: float,
    pool_size: int,
    seed: int,
    gpu_device: int,
    skip_hnsw: bool,
    hardness_field: str = "post",
    tag: str = "match_pdf",
):
    print(f"\n{'=' * 70}\n[{kind}] starting {datetime.now():%H:%M:%S}", flush=True)
    ids, vecs, col, col_name, q_type = load_table(data_dir, kind)
    filters = load_filters(data_dir, kind)
    n, d = vecs.shape
    print(f"  N={n:,} d={d} filters={len(filters)} col={col_name}", flush=True)

    rng = np.random.RandomState(seed)
    # Candidate query pool: sample from base (Algorithm 2 qv sampling)
    pool_size = min(pool_size, n)
    q_idx = rng.choice(n, pool_size, replace=False)
    q_idx.sort()
    q_vecs = vecs[q_idx]
    q_ids = ids[q_idx]

    # Exact full-index for alpha d0 and GLS radius
    print("  Building full Flat-L2 index...", flush=True)
    full_index = build_flat_l2(vecs, device=gpu_device if has_gpu() else None)

    print(f"  Computing d0 (K={K_ALPHA}) and full {K_GLS}-NN for GLS...", flush=True)
    d0 = knn_kth_excluding_self(full_index, q_vecs, q_idx, K_ALPHA, id_map=None)
    # GPU FAISS caps k at 2048; exclude self from the returned neighbourhood.
    k_search = min(K_GLS, 2048)
    D_full, I_full = full_index.search(
        np.ascontiguousarray(q_vecs, dtype=np.float32), k_search
    )
    full_nn_ids = np.full((pool_size, K_GLS), -1, dtype=np.int64)
    for i in range(pool_size):
        picked = []
        for j in range(I_full.shape[1]):
            gid = int(I_full[i, j])
            if gid < 0 or gid == int(q_idx[i]):
                continue
            picked.append(gid)
            if len(picked) == K_GLS:
                break
        full_nn_ids[i, : len(picked)] = picked
    # effective neighbourhood size per query (usually K_GLS-1 when self is in top-k)
    nn_counts = (full_nn_ids >= 0).sum(axis=1)
    print(
        f"  GLS neighbourhood sizes: min={nn_counts.min()} med={int(np.median(nn_counts))} "
        f"max={nn_counts.max()}",
        flush=True,
    )
    del D_full, I_full

    # Optional HNSW for H_fetch
    hnsw = None
    if not skip_hnsw:
        print("  Building HNSW for H_fetch...", flush=True)
        t0 = time.time()
        hnsw = build_hnsw(vecs)
        print(f"  HNSW built in {time.time() - t0:.1f}s", flush=True)

    # Evaluate every (query, filter) pair in the pool
    records = []  # flat list of candidate dicts
    print("  Evaluating filter x query pool...", flush=True)
    for fi, (fstr, sel_nominal) in enumerate(filters):
        fidx = filtered_indices(col, fstr)
        n_f = len(fidx)
        sel = n_f / n
        if n_f < K_ALPHA + 1:
            print(f"    skip {fstr}: |F|={n_f}", flush=True)
            continue
        t0 = time.time()
        filt_dev = None
        if has_gpu():
            filt_dev = (gpu_device + 1) % faiss.get_num_gpus()
        findex = build_flat_l2(vecs[fidx], device=filt_dev)

        df = knn_kth_excluding_self(findex, q_vecs, q_idx, K_ALPHA, id_map=fidx)
        alpha = compute_alpha_batch(d0, df, n, n_f)

        # Exact GLS from precomputed full neighbourhood
        fmask = np.zeros(n, dtype=bool)
        fmask[fidx] = True
        gls, sigma_l = gls_from_full_neighbors(full_nn_ids, fmask, sel)

        # true top-K_TRUE within filter (for storage)
        k_true = min(K_TRUE, n_f)
        D_f, I_f = findex.search(
            np.ascontiguousarray(q_vecs, dtype=np.float32), k_true + 1
        )
        neigh_ids = np.empty((pool_size, k_true), dtype=object)
        neigh_dists = np.empty((pool_size, k_true), dtype=np.float64)
        for i in range(pool_size):
            picked = []
            pdists = []
            for j in range(I_f.shape[1]):
                loc = int(I_f[i, j])
                if loc < 0:
                    continue
                gid = int(fidx[loc])
                if gid == int(q_idx[i]):
                    continue
                picked.append(ids[gid])
                pdists.append(float(np.sqrt(max(D_f[i, j], 0.0))))
                if len(picked) == k_true:
                    break
            while len(picked) < k_true:
                picked.append("")
                pdists.append(np.nan)
            neigh_ids[i] = picked
            neigh_dists[i] = pdists

        if hnsw is not None:
            h_fetch = compute_h_fetch_batch(hnsw, q_vecs, alpha)
            post = alpha * h_fetch / 100.0
        else:
            # Portable surrogate: H_fetch ~= 1 (latency-free). Post = alpha/100.
            # Still useful for ranking within a machine-independent scale.
            h_fetch = np.ones(pool_size, dtype=np.float64)
            post = alpha * h_fetch / 100.0

        for i in range(pool_size):
            if not np.isfinite(post[i]) or not np.isfinite(alpha[i]):
                continue
            records.append(
                {
                    "pool_i": i,
                    "q_idx": int(q_idx[i]),
                    "q_id": str(q_ids[i]),
                    "filter": fstr,
                    "filter_id": fi,
                    "selectivity": float(sel),
                    "alpha_hardness": float(alpha[i]),
                    "h_fetch": float(h_fetch[i]),
                    "post_hardness": float(post[i]),
                    "gls_correlation": float(gls[i]),
                    "sigma_l": float(sigma_l[i]),
                    "density": float(df[i] / (d0[i] + 1e-9)) if d0[i] > EPS else 1.0,
                    "neigh_ids": neigh_ids[i],
                    "neigh_dists": neigh_dists[i],
                }
            )
        del findex, D_f, I_f
        print(
            f"    [{fi:02d}] {fstr:<28} sel={sel:.4f}  "
            f"post[med]={np.median(post):.3f}  alpha[med]={np.median(alpha):.2f}  "
            f"gls[med]={np.nanmedian(gls):.3f}  ({time.time() - t0:.1f}s)",
            flush=True,
        )

    del full_index
    if not records:
        raise RuntimeError("No candidate records produced")

    # Match-PDF on chosen hardness field
    field_key = "post_hardness" if hardness_field == "post" else "alpha_hardness"
    hardness = np.array([r[field_key] for r in records], dtype=np.float64)
    bin_edges = np.linspace(h_min, h_max, n_bins + 1)
    target_counts = np.full(n_bins, n_queries // n_bins, dtype=int)
    # distribute remainder
    for i in range(n_queries - int(target_counts.sum())):
        target_counts[i % n_bins] += 1

    print(
        f"  Match-PDF on {field_key}: {len(records):,} candidates, "
        f"target bins={target_counts.tolist()} edges={bin_edges.tolist()}",
        flush=True,
    )
    print(
        f"  Candidate {field_key} percentiles: "
        f"{np.percentile(hardness, [0, 5, 25, 50, 75, 95, 100])}",
        flush=True,
    )

    np.random.seed(seed)
    sel_idx, bins, fill = match_pdf(hardness, bin_edges, target_counts)
    print(
        f"  Selected {len(sel_idx)} queries "
        f"(exact={int((fill == 0).sum())}, closest_fill={int((fill == 1).sum())})",
        flush=True,
    )
    for b in range(n_bins):
        mask = bins == b
        hs = hardness[sel_idx[mask]] if mask.any() else np.array([])
        print(
            f"    bin {b} [{bin_edges[b]:.1f},{bin_edges[b + 1]:.1f}): "
            f"n={int(mask.sum())}  "
            f"h=[{hs.min():.3f},{hs.max():.3f}]" if len(hs) else
            f"    bin {b} [{bin_edges[b]:.1f},{bin_edges[b + 1]:.1f}): n=0",
            flush=True,
        )

    # Materialize selected rows (sort by bin then original order)
    order = np.argsort(bins)
    sel_idx = sel_idx[order]
    bins = bins[order]
    fill = fill[order]
    chosen = [records[i] for i in sel_idx]

    # Attach vectors
    test = np.stack([q_vecs[r["pool_i"]] for r in chosen]).astype(np.float64)
    mids = np.array([r["q_id"] for r in chosen], dtype=object)
    filters_out = np.array([r["filter"] for r in chosen], dtype=object)
    neigh = np.stack([r["neigh_ids"] for r in chosen])
    neigh_d = np.stack([r["neigh_dists"] for r in chosen]).astype(np.float64)

    id_key = "mid" if kind == "movies" else "rid"
    ids_key = "mids" if kind == "movies" else "rids"

    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"{kind}_hcbgen_{tag}.hdf5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("test", data=test)
        f.create_dataset(id_key, data=np.array(mids, dtype=STR_DT))
        f.create_dataset(ids_key, data=np.array(neigh, dtype=STR_DT))
        f.create_dataset("distances", data=neigh_d)
        f.create_dataset("filter", data=np.array(filters_out, dtype=STR_DT))
        f.create_dataset("selectivity", data=np.array([r["selectivity"] for r in chosen]))
        f.create_dataset("post_hardness", data=np.array([r["post_hardness"] for r in chosen]))
        f.create_dataset("alpha_hardness", data=np.array([r["alpha_hardness"] for r in chosen]))
        f.create_dataset("h_scan", data=np.array([r["alpha_hardness"] for r in chosen]))
        f.create_dataset("h_fetch", data=np.array([r["h_fetch"] for r in chosen]))
        f.create_dataset("density", data=np.array([r["density"] for r in chosen]))
        f.create_dataset("gls_correlation", data=np.array([r["gls_correlation"] for r in chosen]))
        f.create_dataset("sigma_l", data=np.array([r["sigma_l"] for r in chosen]))
        f.create_dataset("hardness_bin", data=bins.astype(np.int32))
        f.create_dataset("fill_method", data=fill.astype(np.int32))  # 0 exact, 1 closest
        f.create_dataset("bin_edges", data=bin_edges.astype(np.float64))
        f.attrs["query_type"] = q_type
        f.attrs["algorithm"] = "HCBGen Algorithm 2 Match-PDF"
        f.attrs["hardness_field"] = field_key
        f.attrs["hardness_metric"] = (
            "Post_Hardness = H_scan * H_fetch / 100  "
            "(paper alpha-Hardness; H_scan is intermediate only)"
        )
        f.attrs["k_alpha"] = K_ALPHA
        f.attrs["k_gls"] = K_GLS
        f.attrs["seed"] = seed
        f.attrs["skip_hnsw"] = int(skip_hnsw)
        if skip_hnsw:
            f.attrs["warning"] = (
                "skip_hnsw=1 => H_fetch=1, Post_Hardness=H_scan/100 "
                "(not comparable to paper semi-real histograms)"
            )
    print(f"  Saved {h5_path}", flush=True)

    # CSV meta + correlation stats
    df_out = pd.DataFrame(
        {
            "q_local_id": np.arange(len(chosen)),
            "q_id": mids,
            "query_type": q_type,
            "filter": filters_out,
            "selectivity": [r["selectivity"] for r in chosen],
            "post_hardness": [r["post_hardness"] for r in chosen],
            "alpha_hardness": [r["alpha_hardness"] for r in chosen],
            "h_fetch": [r["h_fetch"] for r in chosen],
            "gls_correlation": [r["gls_correlation"] for r in chosen],
            "sigma_l": [r["sigma_l"] for r in chosen],
            "density": [r["density"] for r in chosen],
            "hardness_bin": bins,
            "fill_method": np.where(fill == 0, "exact", "closest_fill"),
        }
    )
    csv_path = out_dir / f"{kind}_hcbgen_{tag}_meta.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}", flush=True)

    # GLS vs alpha analysis (numpy-only correlations; scipy may be absent)
    def _pearson(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        x = x - x.mean(); y = y - y.mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / den) if den > 0 else float("nan")

    def _spearman(x, y):
        def rank(a):
            order = np.argsort(a)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(a), dtype=float)
            # average ties
            _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
            for i, c in enumerate(counts):
                if c > 1:
                    ranks[inv == i] = ranks[inv == i].mean()
            return ranks
        return _pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))

    rho_s_alpha = _spearman(df_out["alpha_hardness"], df_out["gls_correlation"])
    rho_s_post = _spearman(df_out["post_hardness"], df_out["gls_correlation"])
    rho_p_alpha = _pearson(df_out["alpha_hardness"], df_out["gls_correlation"])
    rho_p_post = _pearson(df_out["post_hardness"], df_out["gls_correlation"])
    analysis = {
        "table": kind,
        "n": len(df_out),
        "control_metric": field_key,
        "spearman_gls_vs_post": float(rho_s_post),
        "spearman_gls_vs_h_scan": float(rho_s_alpha),
        "pearson_gls_vs_post": float(rho_p_post),
        "pearson_gls_vs_h_scan": float(rho_p_alpha),
        "gls_mean": float(df_out["gls_correlation"].mean()),
        "gls_std": float(df_out["gls_correlation"].std()),
        "post_mean": float(df_out["post_hardness"].mean()),
        "post_min": float(df_out["post_hardness"].min()),
        "post_max": float(df_out["post_hardness"].max()),
        "h_scan_mean": float(df_out["alpha_hardness"].mean()),
        "h_fetch_mean": float(df_out["h_fetch"].mean()),
        "density_mean": float(df_out["density"].mean()),
        "exact_fill_frac": float((fill == 0).mean()),
        "bin_counts": {int(b): int((bins == b).sum()) for b in range(n_bins)},
        "bin_post_mean": {
            int(b): float(df_out.loc[df_out["hardness_bin"] == b, "post_hardness"].mean())
            if (df_out["hardness_bin"] == b).any() else None
            for b in range(n_bins)
        },
    }
    with open(out_dir / f"{kind}_gls_vs_post_{tag}.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("  GLS vs Post_Hardness:", json.dumps(analysis, indent=2), flush=True)

    # Scatter plot — Post_Hardness is the paper metric
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sc0 = axes[0].scatter(
            df_out["post_hardness"],
            df_out["gls_correlation"],
            c=df_out["hardness_bin"],
            cmap="viridis",
            s=18,
            alpha=0.75,
            edgecolors="none",
        )
        axes[0].set_xlabel("Post_Hardness (paper α-Hardness)")
        axes[0].set_ylabel("GLS correlation ρ")
        axes[0].set_title(f"{kind}: GLS vs Post_H  (Spearman={rho_s_post:.3f})")
        fig.colorbar(sc0, ax=axes[0], label="hardness bin")

        sc1 = axes[1].scatter(
            df_out["alpha_hardness"],
            df_out["gls_correlation"],
            c=df_out["hardness_bin"],
            cmap="viridis",
            s=18,
            alpha=0.75,
            edgecolors="none",
        )
        axes[1].set_xlabel("H_scan (intermediate, before /100)")
        axes[1].set_ylabel("GLS correlation ρ")
        axes[1].set_title(f"{kind}: GLS vs H_scan  (Spearman={rho_s_alpha:.3f})")
        fig.colorbar(sc1, ax=axes[1], label="hardness bin")
        fig.tight_layout()
        png = out_dir / f"{kind}_gls_vs_post_{tag}.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"  Saved {png}", flush=True)
    except Exception as e:
        print(f"  (plot skipped: {e})", flush=True)

    return analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=REPO_DATA)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--tables", nargs="+", default=["movies"], choices=["movies", "reviews"])
    ap.add_argument("--n_queries", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--h_min", type=float, default=0.0)
    ap.add_argument("--h_max", type=float, default=10.0)
    ap.add_argument("--pool_size", type=int, default=500,
                    help="Candidate query vectors sampled from base per table")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_device", type=int, default=0)
    ap.add_argument("--skip_hnsw", action="store_true",
                    help="DEBUG only: H_fetch=1 so Post=H_scan/100 (NOT paper-comparable)")
    ap.add_argument(
        "--hardness_field",
        choices=["post", "alpha"],
        default="post",
        help="Match-PDF control signal (default: post = paper Post_Hardness)",
    )
    ap.add_argument("--tag", type=str, default="match_pdf",
                    help="Output filename tag")
    args = ap.parse_args()

    if args.hardness_field == "post" and args.skip_hnsw:
        print(
            "WARNING: --skip_hnsw with Post_Hardness control yields H_scan/100 "
            "(no fetch latency). Paper semi-real PDFs include H_fetch.",
            flush=True,
        )

    out_dir = args.out_dir or (args.data_dir / "hard_queries")
    out_dir.mkdir(parents=True, exist_ok=True)

    target = {
        "pdf": "uniform",
        "range": [args.h_min, args.h_max],
        "n_bins": args.n_bins,
        "per_bin": args.n_queries // args.n_bins,
        "n_queries": args.n_queries,
        "hardness_field": args.hardness_field,
        "hardness_metric": "Post_Hardness = H_scan * H_fetch / 100 (HCBGen v5.1 paper score)",
        "h_scan_note": "H_scan (=alpha) is intermediate only; typically ~10-40 before /100",
        "skip_hnsw": bool(args.skip_hnsw),
        "algorithm": "Algorithm 2 Match-PDF with Budgeted Closest-Fill Fallback",
        "filter_sampling": (
            "MoRe adaptation: sample qf from existing MoRe range-filter set "
            "(avg_rating/total_votes thresholds); sample qv from base vectors. "
            "Original Algorithm 1 samples categorical predicates from a base "
            "payload with per-attribute missing_prob (default 0.5 for random "
            "mode; adjusted during regeneration)."
        ),
    }
    with open(out_dir / f"target_pdf_{args.tag}.json", "w") as f:
        json.dump(target, f, indent=2)
    # canonical name for the primary uniform-[0,10] Post_Hardness run
    if args.tag == "match_pdf" and args.hardness_field == "post":
        with open(out_dir / "target_pdf_uniform_0_10.json", "w") as f:
            json.dump(target, f, indent=2)

    print(f"faiss {faiss.__version__}  gpus={faiss.get_num_gpus()}")
    print(f"out_dir={out_dir}")

    summaries = {}
    for kind in args.tables:
        summaries[kind] = process_table(
            kind=kind,
            data_dir=args.data_dir,
            out_dir=out_dir,
            n_queries=args.n_queries,
            n_bins=args.n_bins,
            h_min=args.h_min,
            h_max=args.h_max,
            pool_size=args.pool_size,
            seed=args.seed,
            gpu_device=args.gpu_device,
            skip_hnsw=args.skip_hnsw,
            hardness_field=args.hardness_field,
            tag=args.tag,
        )

    with open(out_dir / f"summary_{args.tag}.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
