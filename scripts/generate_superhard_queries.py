#!/usr/bin/env python3
"""Generate superhard MoRe movies queries by exploiting near-duplicates.

Near-duplicate structure (remakes / same-title clusters) makes d(q, 10-NN | V0)
very small. A filter that drops those neighbours inflates
density = d(q, 10-NN | F) / d(q, 10-NN | V0), which multiplies HCBGen H_scan
and therefore Post_Hardness = H_scan * H_fetch / 100.

Output: data/datasets/MoRe_large/superhard_queries/ (+ stats plots).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    import faiss
except ImportError as e:  # pragma: no cover
    raise SystemExit("Need faiss-gpu (fanns_gpu env)") from e

# Reuse Match-PDF + helpers from the main generator
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_hcbgen_hard_queries import (  # noqa: E402
    K_ALPHA,
    K_GLS,
    K_TRUE,
    STR_DT,
    build_flat_l2,
    build_hnsw,
    compute_alpha_batch,
    compute_h_fetch_batch,
    gls_from_full_neighbors,
    has_gpu,
    knn_kth_excluding_self,
    match_pdf,
)

REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "datasets" / "MoRe_large"
SKIP_TITLES = {
    "pilot",
    "part 1",
    "part 2",
    "part 3",
    "finale",
    "premiere",
    "the end",
    "home",
}


def load_movies(data_dir: Path):
    path = data_dir / "datasets" / "movies_dataset_0.hdf5"
    with h5py.File(path, "r") as f:
        ids = np.array(
            [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["train_mid"][:]],
            dtype=object,
        )
        titles = np.array(
            [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["train_title"][:]],
            dtype=object,
        )
        vecs = np.ascontiguousarray(f["train_mvector"][:], dtype=np.float32)
        ratings = f["train_avgrating"][:].astype(np.float64)
        years = f["train_year"][:].astype(np.float64)
    return ids, titles, vecs, ratings, years


def remake_seed_indices(titles: np.ndarray) -> np.ndarray:
    key = np.array([t.lower().strip() for t in titles], dtype=object)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(key):
        groups[k].append(i)
    seeds = []
    for k, idxs in groups.items():
        if not (4 <= len(idxs) <= 80):
            continue
        if k.startswith("episode #") or k in SKIP_TITLES:
            continue
        seeds.extend(idxs)
    return np.unique(np.asarray(seeds, dtype=np.int64))


def build_filter_bank(ratings: np.ndarray, years: np.ndarray, data_dir: Path):
    """MoRe rating filters + year filters that prune remake neighbourhoods."""
    out = []
    # existing MoRe rating filters
    fpath = data_dir / "filters" / "movies_filters_0.hdf5"
    with h5py.File(fpath, "r") as f:
        for s, flt in zip(f["selectivities"][:], f["filters"][:]):
            flt = flt.decode() if isinstance(flt, (bytes, np.bytes_)) else str(flt)
            if flt.lower() in {"no_filter", "nofilter"}:
                continue
            # parse "avg_rating >= thr"
            thr = float(flt.split(">=")[-1].strip())
            fidx = np.nonzero(ratings >= thr)[0]
            out.append((flt, float(len(fidx) / len(ratings)), fidx, "rating"))
    # extra aggressive rating cuts
    for thr in (9.0, 9.2, 9.5):
        fidx = np.nonzero(ratings >= thr)[0]
        fstr = f"avg_rating >= {thr}"
        if not any(x[0] == fstr for x in out):
            out.append((fstr, float(len(fidx) / len(ratings)), fidx, "rating"))
    # year filters (exclude most remakes outside an era)
    for thr in (1950, 1960, 1970):
        fidx = np.nonzero(years <= thr)[0]
        out.append((f"year <= {thr}", float(len(fidx) / len(years)), fidx, "year"))
    for thr in (2010, 2015, 2020):
        fidx = np.nonzero(years >= thr)[0]
        out.append((f"year >= {thr}", float(len(fidx) / len(years)), fidx, "year"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=REPO_DATA)
    ap.add_argument("--n_queries", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--h_min", type=float, default=0.0)
    ap.add_argument("--h_max", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_device", type=int, default=0)
    ap.add_argument("--d0_min", type=float, default=0.02,
                    help="Min unfiltered 10-NN distance (exclude near-identical dups)")
    ap.add_argument("--d0_max", type=float, default=0.70,
                    help="Max unfiltered 10-NN distance (near-dup neighbourhood)")
    ap.add_argument("--max_candidates", type=int, default=4000)
    ap.add_argument("--easy_pool", type=int, default=800,
                    help="Extra random queries to help fill easy Post_Hardness bins")
    args = ap.parse_args()

    out_dir = args.data_dir / "superhard_queries"
    stats_dir = args.data_dir / "superhard_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"faiss {faiss.__version__} gpus={faiss.get_num_gpus()}", flush=True)
    print(f"[{datetime.now():%H:%M:%S}] loading movies...", flush=True)
    ids, titles, vecs, ratings, years = load_movies(args.data_dir)
    n, d = vecs.shape
    rng = np.random.RandomState(args.seed)

    print("  building full Flat-L2...", flush=True)
    full_index = build_flat_l2(vecs, device=args.gpu_device if has_gpu() else None)

    seeds = remake_seed_indices(titles)
    print(f"  remake seeds: {len(seeds):,}", flush=True)
    Ds, Is = full_index.search(vecs[seeds], K_ALPHA + 1)
    d10 = np.zeros(len(seeds))
    for i in range(len(seeds)):
        dists = []
        for j in range(Is.shape[1]):
            if int(Is[i, j]) == int(seeds[i]):
                continue
            dists.append(float(np.sqrt(max(Ds[i, j], 0.0))))
            if len(dists) == K_ALPHA:
                break
        d10[i] = dists[-1] if len(dists) == K_ALPHA else np.nan
    sweet = (d10 >= args.d0_min) & (d10 <= args.d0_max) & np.isfinite(d10)
    cand = seeds[sweet]
    d0_cand = d10[sweet]
    print(
        f"  near-dup candidates (d0 in [{args.d0_min},{args.d0_max}]): {len(cand):,}  "
        f"d0 med={np.median(d0_cand):.3f}",
        flush=True,
    )
    if len(cand) > args.max_candidates:
        # prefer smaller d0 (tighter clusters)
        take = np.argsort(d0_cand)[: args.max_candidates]
        cand, d0_cand = cand[take], d0_cand[take]

    # easy random pool for low Post_Hardness bins
    easy = rng.choice(n, size=min(args.easy_pool, n), replace=False)
    easy = np.setdiff1d(easy, cand, assume_unique=False)
    pool = np.concatenate([cand, easy])
    is_neardup = np.concatenate(
        [np.ones(len(cand), dtype=bool), np.zeros(len(easy), dtype=bool)]
    )
    print(f"  total query pool: {len(pool):,} ({len(cand)} near-dup + {len(easy)} easy)", flush=True)

    q_vecs = vecs[pool]
    d0 = knn_kth_excluding_self(full_index, q_vecs, pool, K_ALPHA, id_map=None)

    # GLS neighbourhoods
    print(f"  computing GLS {K_GLS}-NN...", flush=True)
    k_search = min(K_GLS, 2048)
    D_full, I_full = full_index.search(np.ascontiguousarray(q_vecs, dtype=np.float32), k_search)
    full_nn_ids = np.full((len(pool), K_GLS), -1, dtype=np.int64)
    for i in range(len(pool)):
        picked = []
        for j in range(I_full.shape[1]):
            gid = int(I_full[i, j])
            if gid < 0 or gid == int(pool[i]):
                continue
            picked.append(gid)
            if len(picked) == K_GLS:
                break
        full_nn_ids[i, : len(picked)] = picked
    del D_full, I_full

    print("  building HNSW for H_fetch...", flush=True)
    t0 = time.time()
    hnsw = build_hnsw(vecs)
    print(f"  HNSW in {time.time() - t0:.1f}s", flush=True)

    filters = build_filter_bank(ratings, years, args.data_dir)
    print(f"  filters: {len(filters)}", flush=True)

    records = []
    for fi, (fstr, sel, fidx, fkind) in enumerate(filters):
        if len(fidx) < K_ALPHA + 1:
            continue
        t1 = time.time()
        filt_dev = (args.gpu_device + 1) % faiss.get_num_gpus() if has_gpu() else None
        findex = build_flat_l2(vecs[fidx], device=filt_dev)
        df = knn_kth_excluding_self(findex, q_vecs, pool, K_ALPHA, id_map=fidx)
        alpha = compute_alpha_batch(d0, df, n, len(fidx))
        density = df / (np.maximum(d0, 1e-12))
        # zero-d0 guard matching HCBGen: density -> 1
        density = np.where(d0 <= 1e-12, 1.0, density)

        fmask = np.zeros(n, dtype=bool)
        fmask[fidx] = True
        gls, sigma_l = gls_from_full_neighbors(full_nn_ids, fmask, sel)

        k_true = min(K_TRUE, len(fidx))
        D_f, I_f = findex.search(np.ascontiguousarray(q_vecs, dtype=np.float32), k_true + 1)
        neigh_ids = np.empty((len(pool), k_true), dtype=object)
        neigh_dists = np.empty((len(pool), k_true), dtype=np.float64)
        for i in range(len(pool)):
            picked, pdists = [], []
            for j in range(I_f.shape[1]):
                loc = int(I_f[i, j])
                if loc < 0:
                    continue
                gid = int(fidx[loc])
                if gid == int(pool[i]):
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

        h_fetch = compute_h_fetch_batch(hnsw, q_vecs, alpha)
        post = alpha * h_fetch / 100.0

        for i in range(len(pool)):
            if not (np.isfinite(post[i]) and np.isfinite(alpha[i])):
                continue
            records.append(
                {
                    "pool_i": i,
                    "q_idx": int(pool[i]),
                    "q_id": str(ids[pool[i]]),
                    "title": str(titles[pool[i]]),
                    "near_dup_seed": bool(is_neardup[i]),
                    "filter": fstr,
                    "filter_kind": fkind,
                    "filter_id": fi,
                    "selectivity": float(sel),
                    "alpha_hardness": float(alpha[i]),
                    "h_fetch": float(h_fetch[i]),
                    "post_hardness": float(post[i]),
                    "gls_correlation": float(gls[i]),
                    "sigma_l": float(sigma_l[i]),
                    "density": float(density[i]),
                    "d0": float(d0[i]),
                    "df": float(df[i]),
                    "neigh_ids": neigh_ids[i],
                    "neigh_dists": neigh_dists[i],
                }
            )
        del findex, D_f, I_f
        dens_nd = density[is_neardup]
        print(
            f"    [{fi:02d}] {fstr:<22} sel={sel:.4f}  "
            f"post[med]={np.median(post):.3f}  dens_neardup[p90]={np.nanpercentile(dens_nd,90):.2f}  "
            f"post_neardup[p90]={np.nanpercentile(post[is_neardup],90):.2f}  ({time.time()-t1:.1f}s)",
            flush=True,
        )

    del full_index
    if not records:
        raise RuntimeError("no records")

    hardness = np.array([r["post_hardness"] for r in records], dtype=np.float64)
    dens_all = np.array([r["density"] for r in records])
    print(
        f"\n  candidates={len(records):,}  Post_H p[{np.percentile(hardness,[0,50,95,100])}]  "
        f"density p[{np.percentile(dens_all,[50,90,99,100])}]",
        flush=True,
    )

    bin_edges = np.linspace(args.h_min, args.h_max, args.n_bins + 1)
    target_counts = np.full(args.n_bins, args.n_queries // args.n_bins, dtype=int)
    for i in range(args.n_queries - int(target_counts.sum())):
        target_counts[i % args.n_bins] += 1

    np.random.seed(args.seed)
    # Prefer near-dup high-density records when breaking ties: shuffle but
    # Match-PDF already randomizes; optionally upsample by sorting leftovers hardest-first (already in match_pdf).
    sel_idx, bins, fill = match_pdf(hardness, bin_edges, target_counts)
    print(
        f"  selected {len(sel_idx)}  exact={int((fill==0).sum())} closest={int((fill==1).sum())}",
        flush=True,
    )
    for b in range(args.n_bins):
        m = bins == b
        hs = hardness[sel_idx[m]]
        ds = dens_all[sel_idx[m]]
        print(
            f"    bin {b} [{bin_edges[b]:.1f},{bin_edges[b+1]:.1f}): n={int(m.sum())}  "
            f"post=[{hs.min():.3f},{hs.max():.3f}] dens_med={np.median(ds):.2f}",
            flush=True,
        )

    order = np.argsort(bins)
    sel_idx, bins, fill = sel_idx[order], bins[order], fill[order]
    chosen = [records[i] for i in sel_idx]

    test = np.stack([q_vecs[r["pool_i"]] for r in chosen]).astype(np.float64)
    mids = np.array([r["q_id"] for r in chosen], dtype=object)
    filters_out = np.array([r["filter"] for r in chosen], dtype=object)
    neigh = np.stack([r["neigh_ids"] for r in chosen])
    neigh_d = np.stack([r["neigh_dists"] for r in chosen]).astype(np.float64)

    h5_path = out_dir / "movies_hcbgen_superhard.hdf5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("test", data=test)
        f.create_dataset("mid", data=np.array(mids, dtype=STR_DT))
        f.create_dataset("mids", data=np.array(neigh, dtype=STR_DT))
        f.create_dataset("distances", data=neigh_d)
        f.create_dataset("filter", data=np.array(filters_out, dtype=STR_DT))
        f.create_dataset("title", data=np.array([r["title"] for r in chosen], dtype=STR_DT))
        f.create_dataset("selectivity", data=np.array([r["selectivity"] for r in chosen]))
        f.create_dataset("post_hardness", data=np.array([r["post_hardness"] for r in chosen]))
        f.create_dataset("alpha_hardness", data=np.array([r["alpha_hardness"] for r in chosen]))
        f.create_dataset("h_scan", data=np.array([r["alpha_hardness"] for r in chosen]))
        f.create_dataset("h_fetch", data=np.array([r["h_fetch"] for r in chosen]))
        f.create_dataset("density", data=np.array([r["density"] for r in chosen]))
        f.create_dataset("d0", data=np.array([r["d0"] for r in chosen]))
        f.create_dataset("df", data=np.array([r["df"] for r in chosen]))
        f.create_dataset("gls_correlation", data=np.array([r["gls_correlation"] for r in chosen]))
        f.create_dataset("sigma_l", data=np.array([r["sigma_l"] for r in chosen]))
        f.create_dataset("near_dup_seed", data=np.array([r["near_dup_seed"] for r in chosen], dtype=np.uint8))
        f.create_dataset("hardness_bin", data=bins.astype(np.int32))
        f.create_dataset("fill_method", data=fill.astype(np.int32))
        f.create_dataset("bin_edges", data=bin_edges.astype(np.float64))
        f.attrs["query_type"] = "flex_movies_sim"
        f.attrs["algorithm"] = "HCBGen Algorithm 2 Match-PDF + near-dup density exploit"
        f.attrs["hardness_metric"] = "Post_Hardness = H_scan * H_fetch / 100"
        f.attrs["d0_min"] = args.d0_min
        f.attrs["d0_max"] = args.d0_max
        f.attrs["seed"] = args.seed
    print(f"  saved {h5_path}", flush=True)

    df_out = pd.DataFrame(
        {
            "q_local_id": np.arange(len(chosen)),
            "q_id": mids,
            "title": [r["title"] for r in chosen],
            "near_dup_seed": [r["near_dup_seed"] for r in chosen],
            "filter": filters_out,
            "filter_kind": [r["filter_kind"] for r in chosen],
            "selectivity": [r["selectivity"] for r in chosen],
            "post_hardness": [r["post_hardness"] for r in chosen],
            "alpha_hardness": [r["alpha_hardness"] for r in chosen],
            "h_fetch": [r["h_fetch"] for r in chosen],
            "density": [r["density"] for r in chosen],
            "d0": [r["d0"] for r in chosen],
            "df": [r["df"] for r in chosen],
            "gls_correlation": [r["gls_correlation"] for r in chosen],
            "sigma_l": [r["sigma_l"] for r in chosen],
            "hardness_bin": bins,
            "fill_method": np.where(fill == 0, "exact", "closest_fill"),
        }
    )
    csv_path = out_dir / "movies_hcbgen_superhard_meta.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}", flush=True)

    # ---- stats / plots ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _spearman(x, y):
        def rank(a):
            a = np.asarray(a, float)
            order = np.argsort(a)
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(len(a))
            _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
            for i, cc in enumerate(c):
                if cc > 1:
                    r[inv == i] = r[inv == i].mean()
            return r

        rx, ry = rank(x), rank(y)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / den) if den > 0 else float("nan")

    analysis = {
        "n": len(df_out),
        "exact_fill_frac": float((fill == 0).mean()),
        "post_hardness": {
            "min": float(df_out.post_hardness.min()),
            "max": float(df_out.post_hardness.max()),
            "mean": float(df_out.post_hardness.mean()),
            "median": float(df_out.post_hardness.median()),
        },
        "density": {
            "min": float(df_out.density.min()),
            "max": float(df_out.density.max()),
            "mean": float(df_out.density.mean()),
            "median": float(df_out.density.median()),
            "p90": float(df_out.density.quantile(0.9)),
        },
        "near_dup_frac": float(df_out.near_dup_seed.mean()),
        "spearman_post_vs_density": _spearman(df_out.post_hardness, df_out.density),
        "spearman_post_vs_selectivity": _spearman(df_out.post_hardness, df_out.selectivity),
        "spearman_post_vs_gls": _spearman(df_out.post_hardness, df_out.gls_correlation),
        "spearman_gls_vs_density": _spearman(df_out.gls_correlation, df_out.density),
        "bin_post_mean": {
            int(b): float(df_out.loc[df_out.hardness_bin == b, "post_hardness"].mean())
            for b in range(args.n_bins)
        },
        "bin_density_mean": {
            int(b): float(df_out.loc[df_out.hardness_bin == b, "density"].mean())
            for b in range(args.n_bins)
        },
    }
    with open(out_dir / "movies_superhard_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)
    with open(stats_dir / "movies_superhard_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes[0, 0]
    ax.hist(df_out.post_hardness, bins=30, color="#264653", edgecolor="white")
    ax.axvline(df_out.post_hardness.median(), color="#e76f51", ls="--",
               label=f"median={df_out.post_hardness.median():.2f}")
    ax.set_xlabel("Post_Hardness"); ax.set_ylabel("count")
    ax.set_title("Post_Hardness (superhard)"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    edges = np.linspace(args.h_min, args.h_max, args.n_bins + 1)
    exact = df_out.fill_method == "exact"
    ax.hist(
        [df_out.loc[exact, "post_hardness"], df_out.loc[~exact, "post_hardness"]],
        bins=edges,
        stacked=True,
        color=["#2a9d8f", "#e76f51"],
        label=["exact", "closest-fill"],
        edgecolor="white",
    )
    ax.set_xlabel("Post_Hardness"); ax.set_title(f"Match-PDF [{args.h_min},{args.h_max}] exact={exact.mean():.0%}")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    data = [df_out.loc[df_out.hardness_bin == b, "post_hardness"].values for b in range(args.n_bins)]
    bp = ax.boxplot(data, positions=range(args.n_bins), patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        has_ex = (df_out.loc[df_out.hardness_bin == i, "fill_method"] == "exact").any()
        patch.set_facecolor("#2a9d8f" if has_ex else "#e76f51")
        patch.set_alpha(0.75)
    ax.set_xlabel("bin"); ax.set_ylabel("Post_Hardness"); ax.set_title("Post_H by bin")

    ax = axes[1, 0]
    ax.hist(df_out.density, bins=40, color="#e9c46a", edgecolor="white")
    ax.axvline(1.0, color="red", ls="--")
    ax.set_xlabel("density"); ax.set_title(f"density (med={df_out.density.median():.2f}, max={df_out.density.max():.1f})")

    ax = axes[1, 1]
    sc = ax.scatter(
        df_out.selectivity,
        df_out.post_hardness,
        c=df_out.density,
        cmap="coolwarm",
        s=14,
        alpha=0.75,
        vmin=1.0,
        vmax=max(2.0, float(df_out.density.quantile(0.95))),
    )
    ax.set_xlabel("σ_g"); ax.set_ylabel("Post_Hardness"); ax.set_title("Post_H vs σ_g (color=density)")
    fig.colorbar(sc, ax=ax, label="density")

    ax = axes[1, 2]
    sc = ax.scatter(
        df_out.density,
        df_out.post_hardness,
        c=df_out.hardness_bin,
        cmap="viridis",
        s=14,
        alpha=0.75,
    )
    ax.set_xlabel("density"); ax.set_ylabel("Post_Hardness"); ax.set_title("Density drives superhard Post_H")
    fig.colorbar(sc, ax=ax, label="bin")

    fig.suptitle("MoRe movies — near-dup density exploit (superhard)", y=1.01)
    fig.tight_layout()
    fig.savefig(stats_dir / "movies_superhard_histograms.png", dpi=150, bbox_inches="tight")
    fig.savefig(stats_dir / "movies_superhard_histograms.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "movies_superhard_histograms.png", dpi=150, bbox_inches="tight")

    for col, fname, color in [
        ("post_hardness", "movies_post_hardness_hist.png", "#264653"),
        ("density", "movies_density_hist.png", "#e9c46a"),
        ("gls_correlation", "movies_gls_hist.png", "#6d597a"),
        ("alpha_hardness", "movies_h_scan_hist.png", "#457b9d"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_out[col], bins=30, color=color, edgecolor="white")
        ax.set_xlabel(col); ax.set_ylabel("count"); ax.set_title(col)
        if col == "density":
            ax.axvline(1, color="red", ls="--")
        fig.tight_layout()
        fig.savefig(stats_dir / fname, dpi=140)
        plt.close()

    # GLS vs Post scatter
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(
        df_out.post_hardness,
        df_out.gls_correlation,
        c=df_out.density,
        cmap="coolwarm",
        s=16,
        alpha=0.75,
    )
    ax.set_xlabel("Post_Hardness"); ax.set_ylabel("GLS ρ")
    ax.set_title(f"GLS vs Post_H (Spearman={analysis['spearman_post_vs_gls']:.3f})")
    fig.colorbar(sc, ax=ax, label="density")
    fig.tight_layout()
    fig.savefig(out_dir / "movies_gls_vs_post_superhard.png", dpi=140)
    fig.savefig(stats_dir / "movies_gls_vs_post_superhard.png", dpi=140)
    plt.close()

    readme = f"""# MoRe superhard queries (near-dup density exploit)

Exploits remakes / same-title near-duplicates: small `d0 = d(q,10-NN|V0)` with a
filter that removes those neighbours → large `density = df/d0` → large
`Post_Hardness = H_scan * H_fetch / 100`.

## Results (n={len(df_out)})

- Post_Hardness: min={analysis['post_hardness']['min']:.3f} med={analysis['post_hardness']['median']:.3f} max={analysis['post_hardness']['max']:.3f}
- density: med={analysis['density']['median']:.2f} p90={analysis['density']['p90']:.2f} max={analysis['density']['max']:.2f}
- near_dup_seed fraction: {analysis['near_dup_frac']:.0%}
- Match-PDF exact fill: {analysis['exact_fill_frac']:.0%}
- Spearman(Post, density)={analysis['spearman_post_vs_density']:.3f}
- Spearman(Post, GLS)={analysis['spearman_post_vs_gls']:.3f}

## Files

- `movies_hcbgen_superhard.hdf5` / `_meta.csv`
- `movies_superhard_histograms.png`
- `movies_gls_vs_post_superhard.png`
- See also `../superhard_stats/`
"""
    (out_dir / "README.md").write_text(readme)
    (stats_dir / "README.md").write_text(readme)
    df_out.to_csv(stats_dir / "movies_superhard_with_parts.csv", index=False)

    print(json.dumps(analysis, indent=2), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
