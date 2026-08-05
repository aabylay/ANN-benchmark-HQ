#!/usr/bin/env python3
"""Query-optimizer analysis for hard / superhard MoRe packs.

Consumes results under::

    results/MoRe_UPD_large_{hard|superhard}_{movies|reviews}/10/{algo}/...

Joins per-query filter / σ / Post_Hardness / exact GLS from the pack HDF5,
optionally attaches estimated GLS (ρ̂) from a cache CSV, and emits the same
oracle CSVs/plots as the flex analysis plus tertile-binned QPS–recall frontiers.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from query_optimizer_analysis import (  # noqa: E402
    ANN_METHODS,
    BF_METHODS,
    FAISS_ALL_ALGOS,
    FAISS_BUILTIN_ALGOS,
    RECALL_TARGET,
    analyze_hyperparam_recommendations,
    compute_system_oracle_speedup,
    find_best_hyperparams,
    generate_scatter_plots,
    plan_label,
    plot_faiss_all_qps_recall_frontier,
    plot_qps_recall_frontier,
    plot_system_speedup,
    verify_brute_force,
    _plot_qps_recall_frontier_panel,
    _present_algos,
)


def hard_pack_path(root_data: Path, dataset_size: str, table: str, hardness: str) -> Path:
    if hardness == "hard":
        return (
            root_data
            / f"MoRe_{dataset_size}"
            / "hard_queries"
            / f"{table}_hcbgen_match_pdf.hdf5"
        )
    if hardness == "superhard":
        return (
            root_data
            / f"MoRe_{dataset_size}"
            / "superhard_queries"
            / f"{table}_hcbgen_superhard.hdf5"
        )
    raise ValueError(hardness)


def load_train_id_map(root_data: Path, dataset_size: str, table: str) -> dict[str, int]:
    path = root_data / f"MoRe_{dataset_size}" / "datasets" / f"{table}_dataset_0.hdf5"
    with h5py.File(path, "r") as f:
        key = "train_mid" if table == "movies" else "train_rid"
        ids = f[key][:]
    out: dict[str, int] = {}
    for i, x in enumerate(ids):
        s = x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
        out[s] = i
    return out


def load_pack_meta(pack_path: Path, table: str, id_map: dict[str, int], k: int = 10):
    """Return per-query meta + GT neighbor row indices (pack L2 GT as fallback)."""
    with h5py.File(pack_path, "r") as f:
        n = f["test"].shape[0]
        filters = [
            x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["filter"][:]
        ]
        selectivity = f["selectivity"][:].astype(np.float64)
        post_h = f["post_hardness"][:].astype(np.float64)
        gls = f["gls_correlation"][:].astype(np.float64)
        id_key = "mids" if table == "movies" else "rids"
        raw_ids = f[id_key][:, :k]
        gt = np.full((n, k), -1, dtype=np.int64)
        for i in range(n):
            for j in range(k):
                sid = raw_ids[i, j]
                s = sid.decode() if isinstance(sid, (bytes, np.bytes_)) else str(sid)
                gt[i, j] = id_map.get(s, -1)
        Q = np.asarray(f["test"][:], dtype=np.float32)
    return {
        "n": n,
        "filters": filters,
        "selectivity": selectivity,
        "post_hardness": post_h,
        "gls_correlation": gls,
        "gt_neighbors": gt,
        "Q": Q,
    }


def angular_gt_cache_path(root_data: Path, dataset_size: str, table: str, hardness: str, k: int) -> Path:
    return (
        root_data
        / f"MoRe_{dataset_size}"
        / "stats"
        / f"angular_gt_{hardness}_{table}_k{k}.npy"
    )


def ensure_angular_gt(
    root_data: Path,
    dataset_size: str,
    table: str,
    hardness: str,
    pack_meta: dict,
    k: int = 10,
) -> np.ndarray:
    """Exact filtered top-k under angular (IP on L2-normalized vectors).

    Pack GT is L2-based; benchmark runner uses angular, so QO recall must use
    angular ground truth. Cached under stats/ after first computation.
    """
    cache = angular_gt_cache_path(root_data, dataset_size, table, hardness, k)
    if cache.is_file():
        gt = np.load(cache)
        if gt.shape == (pack_meta["n"], k):
            print(f"Loaded angular GT cache {cache}")
            return gt

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        print(f"faiss unavailable ({exc}); falling back to pack L2 GT")
        return pack_meta["gt_neighbors"]

    from ann_benchmarks.attrs import build_attrs_dict, filter_mask_from_attrs
    from ann_benchmarks.runner import load_train_dataset, parse_filter

    print(f"Computing angular filtered GT for {hardness}/{table} (k={k})...")
    X_ids, X_train, X_attrs, _ = load_train_dataset(table, dataset_size)
    attrs = build_attrs_dict(X_attrs, table)
    X = np.ascontiguousarray(X_train, dtype=np.float32)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    Q = np.ascontiguousarray(pack_meta["Q"], dtype=np.float32)
    faiss.normalize_L2(Q)
    gt = np.full((pack_meta["n"], k), -1, dtype=np.int64)
    for i, (q, raw) in enumerate(zip(Q, pack_meta["filters"])):
        ff = parse_filter(raw)
        mask = filter_mask_from_attrs(attrs, ff)
        bitmap = np.packbits(mask, bitorder="little")
        bitmap = np.ascontiguousarray(bitmap, dtype=np.uint8)
        params = faiss.SearchParameters()
        params.sel = faiss.IDSelectorBitmap(bitmap)
        _, I = index.search(q.reshape(1, -1), k, params=params)
        gt[i] = I[0]
        if (i + 1) % 200 == 0:
            print(f"  angular GT {i+1}/{pack_meta['n']}", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, gt)
    print(f"Saved angular GT -> {cache}")
    return gt


def parse_hard_hdf5_path(file: Path) -> tuple[str, dict] | tuple[None, None]:
    """Parse ``.../MoRe_UPD_{size}_{hard|superhard}_{table}/{k}/{algo}/{table}_*.hdf5``."""
    parts = file.parts
    try:
        algo = file.parent.name
        k = int(file.parent.parent.name)
        root_name = file.parent.parent.parent.name
    except (IndexError, ValueError):
        return None, None

    m = re.match(r"MoRe_UPD_(small|medium|large)_(hard|superhard)_(movies|reviews)$", root_name)
    if not m:
        return None, None
    dataset_size, hardness, table_from_root = m.groups()

    patterns = [
        ("hnsw", r"(movies|reviews)_angular_M_(\d+)_efConstruction_(\d+)_(\d+)\.hdf5"),
        ("hnsw", r"(movies|reviews)_angular_(\d+)_M_(\d+)_efConstruction_(\d+)_(\d+)\.hdf5"),
        ("ivf", r"(movies|reviews)_angular_clusters_(\d+)_(\d+)\.hdf5"),
        ("ivf", r"(movies|reviews)_angular_(\d+)_nlist_(\d+)_(\d+)\.hdf5"),
        ("bf", r"(movies|reviews)_angular(?:_\d+)?\.hdf5"),
        ("bf", r"(movies|reviews)_angular_0\.hdf5"),
    ]
    for idx_type, pattern in patterns:
        match = re.match(pattern, file.name)
        if not match:
            continue
        query_type = match.group(1)
        if query_type != table_from_root:
            continue
        params: dict = {
            "query_type": query_type,
            "algo": algo,
            "k": k,
            "hardness": hardness,
            "dataset_size": dataset_size,
            "filter_id": 0,
        }
        if idx_type == "hnsw":
            if match.lastindex == 4:
                _, m_val, ef_c, ef_s = match.groups()
            else:
                _, _, m_val, ef_c, ef_s = match.groups()
            params.update(
                {
                    "m": int(m_val),
                    "ef_construction": int(ef_c),
                    "ef_search": int(ef_s),
                    "hyperparam": int(ef_s),
                    "hyperparam_name": "ef_search",
                }
            )
        elif idx_type == "ivf":
            if "clusters" in pattern:
                _, clusters, probes = match.groups()
            else:
                _, _, clusters, probes = match.groups()
            params.update(
                {
                    "clusters": int(clusters),
                    "probes": int(probes),
                    "hyperparam": int(probes),
                    "hyperparam_name": "probes",
                }
            )
        else:
            params.update({"hyperparam": 0, "hyperparam_name": "none"})
        return idx_type, params
    return None, None


def recall_from_neighbors(gt: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """ID-based recall@k (pack GT uses train row indices)."""
    n = pred.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        true_set = {int(x) for x in gt[i, :k] if int(x) >= 0}
        if not true_set:
            continue
        hits = sum(1 for x in pred[i, :k] if int(x) in true_set and int(x) >= 0)
        out[i] = hits / float(k)
    return out


def build_hard_results_df(
    results_dir: Path,
    root_data: Path,
    dataset_size: str,
    table: str,
    hardness: str,
) -> pd.DataFrame:
    pack_path = hard_pack_path(root_data, dataset_size, table, hardness)
    id_map = load_train_id_map(root_data, dataset_size, table)
    meta = load_pack_meta(pack_path, table, id_map, k=10)
    # Prefer angular GT (matches runner metric); falls back to pack L2 GT.
    gt_neighbors = ensure_angular_gt(root_data, dataset_size, table, hardness, meta, k=10)
    meta["gt_neighbors"] = gt_neighbors

    rows: list[dict] = []
    for file in sorted(results_dir.rglob("*.hdf5")):
        idx_type, params = parse_hard_hdf5_path(file)
        if params is None:
            continue
        if params["query_type"] != table or params["hardness"] != hardness:
            continue
        k = params["k"]
        with h5py.File(file, "r") as f:
            neighbors = f["neighbors"][:]
            times = f["times"][:]
        if neighbors.shape[0] != meta["n"]:
            print(f"Skipping {file}: n_queries mismatch {neighbors.shape[0]} vs {meta['n']}")
            continue
        recalls = recall_from_neighbors(meta["gt_neighbors"], neighbors, k)
        for i, (recall, runtime) in enumerate(zip(recalls, times)):
            prefix = "qm" if table == "movies" else "qr"
            row = {
                "query_id": f"{prefix}{i:04d}",
                "query_id_num": i,
                "query_type": table,
                "filter_id": 0,
                "filter_name": meta["filters"][i],
                "filter_selectivity": float(meta["selectivity"][i]),
                "post_hardness": float(meta["post_hardness"][i]),
                "gls_correlation": float(meta["gls_correlation"][i]),
                "k": k,
                "algorithm": params["algo"],
                "recall": float(recall),
                "runtime": float(runtime),
                "qps": 1.0 / float(runtime) if runtime > 0 else np.nan,
                "hyperparam": params["hyperparam"],
                "hyperparam_name": params["hyperparam_name"],
                "hardness": hardness,
            }
            if idx_type == "hnsw":
                row["m"] = params["m"]
                row["ef_search"] = params["ef_search"]
            elif idx_type == "ivf":
                row["clusters"] = params["clusters"]
                row["probes"] = params["probes"]
            rows.append(row)
    return pd.DataFrame(rows)


def attach_gls_est(df: pd.DataFrame, gls_est_path: Path) -> pd.DataFrame:
    """Attach ρ̂ from ``gls_est_{hardness}_{table}.csv`` on query_id_num."""
    if not gls_est_path.is_file():
        return df
    est = pd.read_csv(gls_est_path)
    if "gls_correlation" not in est.columns and "rho_hat" in est.columns:
        est = est.rename(columns={"rho_hat": "gls_correlation"})
    col = "gls_correlation_est" if "gls_correlation" in est.columns else None
    if col is None:
        return df
    est = est.rename(columns={"gls_correlation": "gls_correlation_est", "q_id": "query_id_num"})
    keep = ["query_id_num", "gls_correlation_est"]
    if "filter" in est.columns:
        keep.append("filter")
    merged = df.merge(est[keep], on="query_id_num", how="left", suffixes=("", "_estfile"))
    return merged


def assign_tertiles(series: pd.Series) -> tuple[pd.Series, dict]:
    """Equal-count tertiles; returns labels + edge metadata."""
    clean = series.astype(float)
    # rank-based split for equal counts even with ties
    ranks = clean.rank(method="first")
    try:
        labels = pd.qcut(ranks, 3, labels=["low", "mid", "high"])
    except ValueError:
        labels = pd.Series(["mid"] * len(series), index=series.index)
    edges = {
        "p33": float(clean.quantile(1 / 3)),
        "p66": float(clean.quantile(2 / 3)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "n": int(clean.notna().sum()),
    }
    for lab in ["low", "mid", "high"]:
        edges[f"n_{lab}"] = int((labels == lab).sum())
    return labels, edges


# Metric → (label, RGB base hue). Tertiles use light / mid / dark shades of that hue.
METRIC_SPECS = [
    ("post_hardness", "Post_Hardness", (0.75, 0.12, 0.12)),          # red
    ("gls_correlation", "exact GLS", (0.12, 0.30, 0.72)),             # blue
    ("gls_correlation_est", "estimated GLS (ρ̂)", (0.12, 0.55, 0.22)),  # green
]

# low → light, mid → medium, high → dark (lerp toward white / black)
_TERTILE_SHADE = {
    "low": 0.55,   # blend toward white
    "mid": 0.0,    # base hue
    "high": -0.45, # blend toward black (negative = darken)
}


def _shade_rgb(base: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    """amount>0 lighten toward white; amount<0 darken toward black."""
    r, g, b = base
    if amount >= 0:
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
    a = -amount
    return (r * (1 - a), g * (1 - a), b * (1 - a))


def _algo_slug(algo: str) -> str:
    return (
        algo.replace("(", "_")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _hp_curve(adf: pd.DataFrame) -> pd.DataFrame:
    return (
        adf.groupby("hyperparam", dropna=False)
        .agg(mean_recall=("recall", "mean"), mean_qps=("qps", "mean"))
        .reset_index()
        .sort_values("mean_recall")
    )


def _attach_tertile(df: pd.DataFrame, metric_col: str) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    if metric_col not in df.columns or df[metric_col].isna().all():
        return None, None
    qmeta = (
        df.groupby("query_id_num", as_index=False)[metric_col]
        .first()
        .dropna(subset=[metric_col])
    )
    labels, edges = assign_tertiles(qmeta[metric_col])
    qmeta = qmeta.assign(tertile=labels.values)
    out = df.merge(qmeta[["query_id_num", "tertile"]], on="query_id_num", how="inner")
    return out, edges


def plot_tertile_qps_recall(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    output_path: Path,
    recall_target: float = RECALL_TARGET,
    algos: list[str] | None = None,
):
    """Legacy 3-panel overview (all algos) by tertiles of one metric."""
    tagged, edges = _attach_tertile(df, metric_col)
    if tagged is None:
        print(f"Skipping tertile plot {output_path.name}: missing {metric_col}")
        return None

    if algos is None:
        algos = _present_algos(
            tagged, [a for a in FAISS_ALL_ALGOS if a in ANN_METHODS] + list(ANN_METHODS)
        )
    algos = _present_algos(tagged, algos)

    # One base color per algorithm; tertiles = light / mid / dark shades.
    algo_bases = {
        "hnsw(faiss)": (0.18, 0.70, 0.40),
        "hnsw(faiss)-post": (0.10, 0.50, 0.28),
        "faiss-ivf": (0.10, 0.30, 0.55),
        "faiss-ivf-post": (0.20, 0.50, 0.80),
        "pgvector": (0.25, 0.45, 0.90),
        "pgvector_ivf": (0.55, 0.30, 0.70),
        "faiss-flat": (0.75, 0.15, 0.15),
        "pgvector_bf": (0.55, 0.10, 0.10),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for algo in algos:
        base = algo_bases.get(algo, (0.3, 0.3, 0.3))
        for tertile in ["low", "mid", "high"]:
            sub = tagged[(tagged["tertile"] == tertile) & (tagged["algorithm"] == algo)]
            if sub.empty:
                continue
            curve = _hp_curve(sub)
            color = _shade_rgb(base, _TERTILE_SHADE[tertile])
            ax.plot(
                curve["mean_recall"],
                curve["mean_qps"],
                "-o",
                markersize=3,
                color=color,
                alpha=0.9,
                label=f"{plan_label(algo)} · {tertile}",
            )
    ax.axvline(recall_target, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Mean recall")
    ax.set_ylabel("Mean QPS")
    ax.set_yscale("log")
    ax.set_title(f"QPS–recall by {metric_label} tertiles (shade: low→light, high→dark)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tertile frontier: {output_path}")
    return edges


def _draw_algo_tertile_panel(
    ax,
    algo: str,
    metric_frames: list,
    recall_target: float,
) -> bool:
    """Draw metric×tertile QPS–recall curves for one algorithm onto ``ax``."""
    any_curve = False
    for _col, label, hue, tagged, _edges in metric_frames:
        adf_all = tagged[tagged["algorithm"] == algo]
        if adf_all.empty:
            continue
        for tertile in ["low", "mid", "high"]:
            sub = adf_all[adf_all["tertile"] == tertile]
            if sub.empty:
                continue
            curve = _hp_curve(sub)
            if curve.empty:
                continue
            color = _shade_rgb(hue, _TERTILE_SHADE[tertile])
            ax.plot(
                curve["mean_recall"],
                curve["mean_qps"],
                "-o",
                markersize=3.5,
                color=color,
                linewidth=1.6,
                alpha=0.95,
                label=f"{label} · {tertile}",
            )
            any_curve = True
            if algo in ANN_METHODS:
                sub_best = find_best_hyperparams(sub, recall_target, algos=[algo])
                if not sub_best.empty:
                    best = sub_best.iloc[0]
                    ax.scatter(
                        [best["mean_recall"]],
                        [best["mean_qps"]],
                        s=70,
                        marker="*",
                        color=color,
                        zorder=5,
                        edgecolors="white",
                        linewidths=0.3,
                    )
            else:
                ax.scatter(
                    [curve["mean_recall"].iloc[-1]],
                    [curve["mean_qps"].iloc[-1]],
                    s=55,
                    marker="s",
                    color=color,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.3,
                )
    ax.axvline(recall_target, color="gray", linestyle="--", alpha=0.55)
    ax.set_xlabel("Mean recall")
    ax.set_ylabel("Mean QPS")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return any_curve


def plot_tertile_qps_recall_per_algorithm(
    df: pd.DataFrame,
    output_dir: Path,
    recall_target: float = RECALL_TARGET,
    algos: list[str] | None = None,
) -> dict[str, dict]:
    """One multi-subplot figure: subplot per algorithm.

    Within each subplot:
      - Post_Hardness  → shades of red   (low light → high dark)
      - exact GLS      → shades of blue
      - estimated GLS  → shades of green

    Writes:
      ``output_dir / qps_recall_tertiles_by_algo.png``
      and per-algo copies under ``qps_recall_tertiles_per_algo/``.
    """
    if algos is None:
        candidate = (
            [a for a in FAISS_ALL_ALGOS if a in ANN_METHODS or a in BF_METHODS]
            + list(ANN_METHODS)
            + list(BF_METHODS)
        )
        seen: set[str] = set()
        algos = []
        for a in candidate:
            if a not in seen:
                seen.add(a)
                algos.append(a)
    algos = _present_algos(df, algos)

    metric_frames: list[tuple[str, str, tuple[float, float, float], pd.DataFrame, dict]] = []
    edges_out: dict[str, dict] = {}
    for col, label, hue in METRIC_SPECS:
        tagged, edges = _attach_tertile(df, col)
        if tagged is None:
            continue
        metric_frames.append((col, label, hue, tagged, edges))
        edges_out[col] = edges

    if not metric_frames or not algos:
        print("No metrics/algorithms available for per-algorithm tertile plots")
        return edges_out

    n = len(algos)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), sharex=False, sharey=False
    )
    axes_flat = np.atleast_1d(axes).ravel()

    legend_handles = None
    legend_labels = None
    for ax, algo in zip(axes_flat, algos):
        _draw_algo_tertile_panel(ax, algo, metric_frames, recall_target)
        ax.set_title(plan_label(algo), fontsize=11)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes_flat[len(algos) :]:
        ax.set_visible(False)

    fig.suptitle(
        "QPS–recall by metric tertiles per algorithm\n"
        "red = Post_Hardness · blue = exact GLS · green = estimated GLS "
        "(shade: low→light, mid, high→dark)",
        fontsize=12,
    )
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    combo_path = output_dir / "qps_recall_tertiles_by_algo.png"
    fig.savefig(combo_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-subplot tertile frontier: {combo_path}")

    out_dir = output_dir / "qps_recall_tertiles_per_algo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for algo in algos:
        mono, mono_ax = plt.subplots(figsize=(6.5, 5.0))
        _draw_algo_tertile_panel(mono_ax, algo, metric_frames, recall_target)
        mono_ax.set_title(
            f"{plan_label(algo)} — QPS–recall by metric tertiles\n"
            "red=Post_H · blue=GLS exact · green=GLS est (shade=low→high)"
        )
        mono_ax.legend(fontsize=8, loc="best")
        mono.tight_layout()
        path = out_dir / f"{_algo_slug(algo)}.png"
        mono.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(mono)
        print(f"Saved per-algo tertile frontier: {path}")

    return edges_out


def run_analysis(
    results_dir: Path,
    root_data: Path,
    dataset_size: str,
    table: str,
    hardness: str,
    output_dir: Path,
    gls_est_path: Path | None,
    recall_target: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building hard results DF: {hardness}/{table} from {results_dir}")
    df = build_hard_results_df(results_dir, root_data, dataset_size, table, hardness)
    if df.empty:
        print("No results found!")
        return {"ok": False, "missing": ["all_query_results.csv"]}

    if gls_est_path is not None:
        df = attach_gls_est(df, gls_est_path)

    df.to_csv(output_dir / "all_query_results.csv", index=False)
    print(f"Saved {output_dir / 'all_query_results.csv'} ({len(df)} rows)")

    # Estimated-GLS variant in the exact layout qo_prototype.py expects:
    # a sibling gls_est/all_query_results.csv whose gls_correlation column
    # holds rho-hat instead of the exact value.
    if "gls_correlation_est" in df.columns and df["gls_correlation_est"].notna().any():
        est_dir = output_dir / "gls_est"
        est_dir.mkdir(parents=True, exist_ok=True)
        df_est = df.copy()
        df_est["gls_correlation"] = df_est["gls_correlation_est"]
        df_est.to_csv(est_dir / "all_query_results.csv", index=False)
        print(f"Saved {est_dir / 'all_query_results.csv'} ({len(df_est)} rows)")

    bf_summary = verify_brute_force(df)
    bf_summary.to_csv(output_dir / "brute_force_recall_check.csv", index=False)
    print(bf_summary.to_string(index=False))

    best_hp = find_best_hyperparams(df, recall_target)
    best_hp.to_csv(output_dir / "best_hyperparameters.csv", index=False)

    hp_sweep, hp_recs = analyze_hyperparam_recommendations(df, recall_target)
    hp_sweep.to_csv(output_dir / "hyperparam_sweep.csv", index=False)
    hp_recs.to_csv(output_dir / "hyperparam_recommendations.csv", index=False)

    speedup_df = compute_system_oracle_speedup(df, best_hp, recall_target)
    speedup_df.to_csv(output_dir / "system_oracle_speedup.csv", index=False)
    plot_system_speedup(speedup_df, output_dir / "system_oracle_speedup.png", recall_target)

    generate_scatter_plots(df, best_hp, output_dir, recall_target, gls_source="exact")
    plot_qps_recall_frontier(df, best_hp, output_dir, recall_target)
    plot_faiss_all_qps_recall_frontier(df, best_hp, output_dir, recall_target)

    # Tertile frontiers — per-metric overview (algo shades) + per-algorithm
    # (metric hue × tertile shade) plots.
    edges_post = plot_tertile_qps_recall(
        df,
        "post_hardness",
        "Post_Hardness",
        output_dir / "qps_recall_by_post_hardness_tertiles.png",
        recall_target,
    )
    if edges_post:
        pd.DataFrame([edges_post]).to_csv(
            output_dir / "tertile_edges_post_hardness.csv", index=False
        )

    edges_gls = plot_tertile_qps_recall(
        df,
        "gls_correlation",
        "exact GLS",
        output_dir / "qps_recall_by_gls_exact_tertiles.png",
        recall_target,
    )
    if edges_gls:
        pd.DataFrame([edges_gls]).to_csv(
            output_dir / "tertile_edges_gls_exact.csv", index=False
        )

    deferred_est = False
    if "gls_correlation_est" in df.columns and df["gls_correlation_est"].notna().any():
        edges_est = plot_tertile_qps_recall(
            df,
            "gls_correlation_est",
            "estimated GLS (ρ̂)",
            output_dir / "qps_recall_by_gls_est_tertiles.png",
            recall_target,
        )
        if edges_est:
            pd.DataFrame([edges_est]).to_csv(
                output_dir / "tertile_edges_gls_est.csv", index=False
            )
    else:
        deferred_est = True
        print(
            f"Deferred estimated-GLS tertiles: missing {gls_est_path}"
            if gls_est_path
            else "Deferred estimated-GLS tertiles: no est path"
        )

    per_algo_edges = plot_tertile_qps_recall_per_algorithm(
        df, output_dir, recall_target
    )
    # Prefer edges from the per-algo pass when present (same splits).
    for col, edges in per_algo_edges.items():
        name = {
            "post_hardness": "tertile_edges_post_hardness.csv",
            "gls_correlation": "tertile_edges_gls_exact.csv",
            "gls_correlation_est": "tertile_edges_gls_est.csv",
        }.get(col)
        if name:
            pd.DataFrame([edges]).to_csv(output_dir / name, index=False)

    required = [
        "all_query_results.csv",
        "best_hyperparameters.csv",
        "best_plan_per_query.csv",
        "qps_recall_by_post_hardness_tertiles.png",
        "qps_recall_by_gls_exact_tertiles.png",
        "qps_recall_tertiles_by_algo.png",
        "qps_recall_tertiles_per_algo",
    ]
    missing = []
    for r in required:
        p = output_dir / r
        if r == "qps_recall_tertiles_per_algo":
            if not p.is_dir() or not any(p.glob("*.png")):
                missing.append(r)
        elif not p.is_file():
            missing.append(r)
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "deferred_gls_est": deferred_est,
        "n_rows": len(df),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardness", required=True, choices=["hard", "superhard"])
    parser.add_argument("--table", required=True, choices=["movies", "reviews"])
    parser.add_argument("--dataset-size", default="large")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Defaults to results/MoRe_UPD_{size}_{hardness}_{table}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to analysis/plots/query_optimizer_{hardness}_{table}",
    )
    parser.add_argument(
        "--gls-est",
        default=None,
        help="Optional ρ̂ CSV (default: data/.../stats/gls_est_{hardness}_{table}.csv)",
    )
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    args = parser.parse_args()

    root_data = ROOT / "data" / "datasets"
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else ROOT / "results" / f"MoRe_UPD_{args.dataset_size}_{args.hardness}_{args.table}"
    )
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "analysis" / "plots" / f"query_optimizer_{args.hardness}_{args.table}"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    gls_est = (
        Path(args.gls_est)
        if args.gls_est
        else root_data
        / f"MoRe_{args.dataset_size}"
        / "stats"
        / f"gls_est_{args.hardness}_{args.table}.csv"
    )
    if not gls_est.is_absolute():
        gls_est = ROOT / gls_est

    status = run_analysis(
        results_dir,
        root_data,
        args.dataset_size,
        args.table,
        args.hardness,
        output_dir,
        gls_est if gls_est.is_file() else None,
        args.recall_target,
    )
    if not status["ok"]:
        print(f"INCOMPLETE: missing {status['missing']}")
        sys.exit(1)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
