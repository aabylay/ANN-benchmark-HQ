#!/usr/bin/env python3
"""
Query optimizer analysis for filtered ANN benchmarks.

1. Build per-query results CSV from HDF5 benchmark outputs.
2. Verify brute-force (faiss-flat, pgvector_bf) recall = 1.0.
3. For each ANN method, pick the hyperparameter that maximizes QPS subject to
   mean recall >= 0.95.
4. For each query, pick the best plan among tuned ANN methods and brute-force.
5. Plot selectivity vs GLS correlation colored by best plan.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ann_benchmarks.plotting.metrics import get_recall_values, knn_threshold

RECALL_TARGET = 0.95
K_DEFAULT = 10

ANN_METHODS = {
    "hnsw(faiss)": {"label": "FAISS HNSW pre", "index": "HNSW", "system": "FAISS", "filter": "pre"},
    "hnsw(faiss)-post": {"label": "FAISS HNSW post", "index": "HNSW", "system": "FAISS", "filter": "post"},
    "faiss-ivf": {"label": "FAISS IVF pre", "index": "IVF", "system": "FAISS", "filter": "pre"},
    "faiss-ivf-post": {"label": "FAISS IVF post", "index": "IVF", "system": "FAISS", "filter": "post"},
    "pgvector": {"label": "pgvector HNSW", "index": "HNSW", "system": "pgvector"},
    "pgvector_ivf": {"label": "pgvector IVF", "index": "IVF", "system": "pgvector"},
}

BF_METHODS = {
    "faiss-flat": {"label": "FAISS BF", "index": "BF", "system": "FAISS"},
    "pgvector_bf": {"label": "pgvector BF", "index": "BF", "system": "pgvector"},
}

SYSTEMS = {
    "FAISS": ["hnsw(faiss)", "hnsw(faiss)-post", "faiss-ivf", "faiss-ivf-post", "faiss-flat"],
    "pgvector": ["pgvector", "pgvector_ivf", "pgvector_bf"],
}

PLAN_COLORS = {
    "FAISS HNSW pre": "#00aa00",
    "FAISS HNSW post": "#66ff66",
    "FAISS IVF pre": "#0000AA",
    "FAISS IVF post": "#6666ff",
    "FAISS BF": "#ff0000",
    "pgvector HNSW": "#00ff00",
    "pgvector IVF": "#0000AA",
    "pgvector BF": "#ff0000",
}


def plan_label(algo: str) -> str:
    if algo in ANN_METHODS:
        return ANN_METHODS[algo]["label"]
    if algo in BF_METHODS:
        return BF_METHODS[algo]["label"]
    return algo


def _plans_at_tuned_hyperparams(
    df: pd.DataFrame, best_hp: pd.DataFrame, algos: list[str]
) -> pd.DataFrame:
    """ANN rows at tuned hyperparams; BF rows have no search param to tune."""
    hp_map = {row["algorithm"]: row["hyperparam"] for _, row in best_hp.iterrows()}
    ann_algos = [a for a in algos if a in ANN_METHODS]
    bf_algos = [a for a in algos if a in BF_METHODS]
    parts: list[pd.DataFrame] = []
    if ann_algos:
        ann = df[df["algorithm"].isin(ann_algos)].copy()
        ann = ann[ann.apply(lambda r: r["hyperparam"] == hp_map[r["algorithm"]], axis=1)]
        parts.append(ann)
    if bf_algos:
        parts.append(df[df["algorithm"].isin(bf_algos)].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def parse_hdf5_path(file: Path) -> tuple[str, dict] | tuple[None, None]:
    """Return (idx_type, params) parsed from an HDF5 result path."""
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

        fid_match = re.match(r"fid(\d+)", file.parent.parent.parent.name)
        filter_id = int(fid_match.group(1))
        k = int(file.parent.parent.name)
        algo = file.parent.name

        if idx_type == "hnsw":
            if match.lastindex == 4:
                query_type, m, ef_construction, ef_search = match.groups()
                dimension = ""
            else:
                query_type, dimension, m, ef_construction, ef_search = match.groups()
            return idx_type, {
                "query_type": query_type,
                "dimension": dimension,
                "m": int(m),
                "ef_construction": int(ef_construction),
                "ef_search": int(ef_search),
                "algo": algo,
                "k": k,
                "filter_id": filter_id,
            }
        if idx_type == "ivf":
            if "clusters" in pattern:
                query_type, clusters, probes = match.groups()
                dimension = ""
            else:
                query_type, dimension, clusters, probes = match.groups()
            return idx_type, {
                "query_type": query_type,
                "dimension": dimension,
                "clusters": int(clusters),
                "probes": int(probes),
                "algo": algo,
                "k": k,
                "filter_id": filter_id,
            }
        if idx_type == "bf":
            query_type = match.group(1)
            return idx_type, {
                "query_type": query_type,
                "algo": algo,
                "k": k,
                "filter_id": filter_id,
            }
    return None, None


def load_selectivities(root_data: Path, dataset_size: str, query_type: str) -> np.ndarray:
    path = root_data / f"MoRe_{dataset_size}" / "filters" / f"{query_type}_filters_0.hdf5"
    with h5py.File(path, "r") as f:
        return f["selectivities"][:]


def load_filter_names(root_data: Path, dataset_size: str, query_type: str) -> list[str]:
    path = root_data / f"MoRe_{dataset_size}" / "filters" / f"{query_type}_filters_0.hdf5"
    with h5py.File(path, "r") as f:
        raw = f["filters"][:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def load_true_distances(
    root_data: Path, dataset_size: str, query_type: str, filter_id: int, k: int
) -> np.ndarray:
    path = (
        root_data
        / f"MoRe_{dataset_size}"
        / "queries"
        / f"queries_flex_{query_type}_sim_0_{filter_id}.hdf5"
    )
    with h5py.File(path, "r") as f:
        return 1 - f["distances"][:, :k]


def process_hdf5_file(
    file: Path,
    params: dict,
    idx_type: str,
    selectivities: np.ndarray,
    true_distances: np.ndarray,
) -> list[dict]:
    with h5py.File(file, "r") as f:
        distances = f["distances"][:]
        times = f["times"][:]

    filter_id = params["filter_id"]
    if filter_id >= len(selectivities):
        return []

    filter_selectivity = float(selectivities[filter_id])
    k = params["k"]
    runs = get_recall_values(true_distances, distances, k, knn_threshold, epsilon=1e-6)
    recalls = runs[2]

    rows = []
    for i, (recall, runtime) in enumerate(zip(recalls, times)):
        query_id = f"q{'m' if params['query_type'] == 'movies' else 'r'}{i:04d}"
        row = {
            "query_id": query_id,
            "query_id_num": i,
            "query_type": params["query_type"],
            "filter_id": filter_id,
            "filter_selectivity": filter_selectivity,
            "k": k,
            "algorithm": params["algo"],
            "recall": recall,
            "runtime": runtime,
            "qps": 1.0 / runtime if runtime > 0 else np.nan,
        }
        if idx_type == "hnsw":
            row["m"] = params["m"]
            row["ef_search"] = params["ef_search"]
            row["hyperparam"] = params["ef_search"]
            row["hyperparam_name"] = "ef_search"
        elif idx_type == "ivf":
            row["clusters"] = params["clusters"]
            row["probes"] = params["probes"]
            row["hyperparam"] = params["probes"]
            row["hyperparam_name"] = "probes"
        else:
            row["hyperparam"] = 0
            row["hyperparam_name"] = "none"
        rows.append(row)
    return rows


def build_results_df(root_results: Path, root_data: Path, dataset_size: str) -> pd.DataFrame:
    rows: list[dict] = []
    cache: dict[tuple, np.ndarray] = {}

    for file in sorted(root_results.rglob("*.hdf5")):
        idx_type, params = parse_hdf5_path(file)
        if params is None:
            continue

        qt = params["query_type"]
        fid = params["filter_id"]
        k = params["k"]
        sel_key = (dataset_size, qt)
        if sel_key not in cache:
            cache[sel_key] = load_selectivities(root_data, dataset_size, qt)

        true_key = (dataset_size, qt, fid, k)
        if true_key not in cache:
            try:
                cache[true_key] = load_true_distances(root_data, dataset_size, qt, fid, k)
            except OSError:
                continue

        try:
            rows.extend(
                process_hdf5_file(file, params, idx_type, cache[sel_key], cache[true_key])
            )
        except Exception as exc:
            print(f"Skipping {file}: {exc}")

    return pd.DataFrame(rows)


def attach_gls_correlation(
    df: pd.DataFrame, filter_stats_path: Path, root_data: Path, dataset_size: str
) -> pd.DataFrame:
    stats = pd.read_csv(filter_stats_path)
    stats = stats.rename(columns={"correlation_NEW": "gls_correlation", "q_id": "query_id_num"})

    filter_lookup: dict[tuple[str, int], tuple[str, float]] = {}
    for qt in df["query_type"].unique():
        names = load_filter_names(root_data, dataset_size, qt)
        sels = load_selectivities(root_data, dataset_size, qt)
        qt_full = f"flex_{qt}_sim"
        for fid, (name, sel) in enumerate(zip(names, sels)):
            filter_lookup[(qt_full, fid)] = (name, float(sel))

    df = df.copy()
    df["query_type_full"] = df["query_type"].map(lambda x: f"flex_{x}_sim")
    df["filter_name"] = df.apply(
        lambda r: filter_lookup.get((r["query_type_full"], r["filter_id"]), (None, None))[0],
        axis=1,
    )

    merged = df.merge(
        stats[["query_id_num", "query_type", "filter", "gls_correlation"]],
        left_on=["query_id_num", "query_type_full", "filter_name"],
        right_on=["query_id_num", "query_type", "filter"],
        how="left",
        suffixes=("", "_stats"),
    )
    merged = merged.drop(columns=["query_type_stats", "filter"], errors="ignore")
    return merged


def find_best_hyperparams(df: pd.DataFrame, recall_target: float = RECALL_TARGET) -> pd.DataFrame:
    ann_df = df[df["algorithm"].isin(ANN_METHODS)].copy()
    grouped = (
        ann_df.groupby(["algorithm", "hyperparam", "hyperparam_name"], dropna=False)
        .agg(mean_recall=("recall", "mean"), mean_qps=("qps", "mean"), n=("recall", "count"))
        .reset_index()
    )

    best_rows = []
    for algo in ANN_METHODS:
        sub = grouped[grouped["algorithm"] == algo]
        # Prefer hyperparameters measured on the full query set (same n as max).
        full_n = sub["n"].max()
        sub_full = sub[sub["n"] == full_n]
        search_pool = sub_full if len(sub_full) else sub

        feasible = search_pool[search_pool["mean_recall"] >= recall_target]
        if len(feasible) == 0:
            pick = search_pool.sort_values(["mean_recall", "mean_qps"], ascending=[False, False]).iloc[0]
            note = f"no config reached target on n={int(pick['n'])} queries; picked highest recall"
        else:
            pick = feasible.sort_values("mean_qps", ascending=False).iloc[0]
            note = f"ok (n={int(pick['n'])} queries)"

        # Also report best config on any partial coverage (may exceed recall on subset)
        partial = sub[sub["n"] < full_n]
        partial_note = ""
        if len(partial):
            p_feas = partial[partial["mean_recall"] >= recall_target]
            if len(p_feas):
                p_best = p_feas.sort_values("mean_qps", ascending=False).iloc[0]
                partial_note = (
                    f" | partial: {p_best['hyperparam_name']}={int(p_best['hyperparam'])} "
                    f"recall={p_best['mean_recall']:.4f} qps={p_best['mean_qps']:.1f} n={int(p_best['n'])}"
                )

        best_rows.append(
            {
                "algorithm": algo,
                "label": ANN_METHODS[algo]["label"],
                "hyperparam": int(pick["hyperparam"]),
                "hyperparam_name": pick["hyperparam_name"],
                "mean_recall": pick["mean_recall"],
                "mean_qps": pick["mean_qps"],
                "n_queries": int(pick["n"]),
                "status": note + partial_note,
            }
        )
    return pd.DataFrame(best_rows)


def _hyperparam_sweep_table(df: pd.DataFrame) -> pd.DataFrame:
    ann_df = df[df["algorithm"].isin(ANN_METHODS)].copy()
    return (
        ann_df.groupby(["algorithm", "hyperparam", "hyperparam_name"], dropna=False)
        .agg(
            mean_recall=("recall", "mean"),
            min_recall=("recall", "min"),
            mean_qps=("qps", "mean"),
            mean_runtime_ms=("runtime", lambda s: s.mean() * 1000),
            n=("recall", "count"),
        )
        .reset_index()
        .sort_values(["algorithm", "hyperparam"])
    )


def analyze_hyperparam_recommendations(
    df: pd.DataFrame, recall_target: float = RECALL_TARGET
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full sweep table plus per-algorithm guidance on which hyperparams reach recall target.

    Returns:
        sweep_table: every (algorithm, hyperparam) with recall/QPS stats
        recommendations: min hyperparam for recall, best-QPS at recall, eval suggestions
    """
    sweep = _hyperparam_sweep_table(df)

    rec_rows = []
    for algo in ANN_METHODS:
        sub = sweep[sweep["algorithm"] == algo].copy()
        hp_name = sub["hyperparam_name"].iloc[0] if len(sub) else "hyperparam"
        full_n = sub["n"].max()
        sub_full = sub[sub["n"] == full_n].sort_values("hyperparam")
        sub_partial = sub[sub["n"] < full_n].sort_values("hyperparam")

        feasible_full = sub_full[sub_full["mean_recall"] >= recall_target]
        feasible_partial = sub_partial[sub_partial["mean_recall"] >= recall_target]

        if len(feasible_full):
            min_row = feasible_full.iloc[0]
            best_qps_row = feasible_full.sort_values("mean_qps", ascending=False).iloc[0]
            min_hp = int(min_row["hyperparam"])
            best_qps_hp = int(best_qps_row["hyperparam"])
            reaches_target = "yes (full coverage)"
            eval_run = (
                f"Use {hp_name}>={min_hp} for recall≥{recall_target}; "
                f"{hp_name}={best_qps_hp} for best QPS at recall≥{recall_target}"
            )
        elif len(feasible_partial):
            min_row = feasible_partial.iloc[0]
            best_qps_row = feasible_partial.sort_values("mean_qps", ascending=False).iloc[0]
            min_hp = int(min_row["hyperparam"])
            best_qps_hp = int(best_qps_row["hyperparam"])
            max_full = sub_full.sort_values("hyperparam").iloc[-1]
            reaches_target = f"no on full data (max {hp_name}={int(max_full['hyperparam'])} → recall={max_full['mean_recall']:.4f})"
            eval_run = (
                f"Re-run with higher {hp_name} on full workload; "
                f"partial n={int(min_row['n'])} reached target at {hp_name}={min_hp}"
            )
        else:
            min_hp = np.nan
            best_qps_hp = np.nan
            if len(sub_full):
                max_row = sub_full.sort_values("mean_recall", ascending=False).iloc[0]
                reaches_target = f"no (best full: {hp_name}={int(max_row['hyperparam'])}, recall={max_row['mean_recall']:.4f})"
                next_hp = int(max_row["hyperparam"]) * 2
                eval_run = f"Extend sweep: run {hp_name} > {int(max_row['hyperparam'])} (e.g. {next_hp})"
            else:
                reaches_target = "no data"
                eval_run = "no data"

        rec_rows.append(
            {
                "algorithm": algo,
                "label": ANN_METHODS[algo]["label"],
                "hyperparam_name": hp_name,
                "recall_target": recall_target,
                "reaches_target_full_coverage": reaches_target,
                "min_hyperparam_for_recall": min_hp,
                "min_hyperparam_mean_recall": feasible_full.iloc[0]["mean_recall"] if len(feasible_full) else (
                    feasible_partial.iloc[0]["mean_recall"] if len(feasible_partial) else np.nan
                ),
                "best_qps_hyperparam_at_recall": best_qps_hp,
                "best_qps_at_recall": feasible_full.sort_values("mean_qps", ascending=False).iloc[0]["mean_qps"]
                if len(feasible_full)
                else (
                    feasible_partial.sort_values("mean_qps", ascending=False).iloc[0]["mean_qps"]
                    if len(feasible_partial)
                    else np.nan
                ),
                "recommended_eval_hyperparams": eval_run,
            }
        )

    return sweep, pd.DataFrame(rec_rows)


def _primary_algo(ann_algos: list[str], index_type: str, filter_mode: str = "pre") -> str:
    """Pick a representative algorithm for baseline HNSW/IVF comparisons."""
    matches = [a for a in ann_algos if ANN_METHODS[a]["index"] == index_type]
    preferred = [a for a in matches if ANN_METHODS[a].get("filter") == filter_mode]
    if preferred:
        return preferred[0]
    if matches:
        return matches[0]
    raise StopIteration(f"No {index_type} algorithm in {ann_algos}")


def _query_runtimes_for_system(
    df: pd.DataFrame,
    best_hp: pd.DataFrame,
    system_algos: list[str],
    recall_target: float,
) -> pd.DataFrame:
    """Per query: runtime/recall/qps for HNSW and IVF in a system at tuned hyperparams."""
    ann_algos = [a for a in system_algos if a in ANN_METHODS]
    ann = _plans_at_tuned_hyperparams(df, best_hp, ann_algos)
    if ann.empty:
        return pd.DataFrame()

    hnsw_algo = _primary_algo(ann_algos, "HNSW")
    ivf_algo = _primary_algo(ann_algos, "IVF")

    keys = ["query_id", "query_type", "filter_id"]
    records = []
    for key_vals, group in ann.groupby(keys):
        hnsw = group[group["algorithm"] == hnsw_algo]
        ivf = group[group["algorithm"] == ivf_algo]
        if hnsw.empty or ivf.empty:
            continue
        h = hnsw.iloc[0]
        v = ivf.iloc[0]

        feasible = group[group["recall"] >= recall_target]
        if len(feasible):
            oracle = feasible.sort_values("qps", ascending=False).iloc[0]
        else:
            oracle = group.sort_values(["recall", "qps"], ascending=[False, False]).iloc[0]

        records.append(
            {
                "query_id": key_vals[0],
                "query_type": key_vals[1],
                "filter_id": key_vals[2],
                "hnsw_runtime": h["runtime"],
                "hnsw_qps": h["qps"],
                "hnsw_recall": h["recall"],
                "ivf_runtime": v["runtime"],
                "ivf_qps": v["qps"],
                "ivf_recall": v["recall"],
                "oracle_runtime": oracle["runtime"],
                "oracle_qps": oracle["qps"],
                "oracle_recall": oracle["recall"],
                "oracle_plan": plan_label(oracle["algorithm"]),
            }
        )
    return pd.DataFrame(records)


def compute_system_oracle_speedup(
    df: pd.DataFrame,
    best_hp: pd.DataFrame,
    recall_target: float = RECALL_TARGET,
) -> pd.DataFrame:
    """
    Speedup from picking the best in-system plan (HNSW vs IVF) per query vs always
    using one index type. Uses tuned hyperparameters for each algorithm.
    """
    summary_rows = []
    for system, algos in SYSTEMS.items():
        ann_algos = [a for a in algos if a in ANN_METHODS]
        hnsw_algo = _primary_algo(ann_algos, "HNSW")
        ivf_algo = _primary_algo(ann_algos, "IVF")
        hnsw_label = plan_label(hnsw_algo)
        ivf_label = plan_label(ivf_algo)

        per_query = _query_runtimes_for_system(df, best_hp, algos, recall_target)
        if per_query.empty:
            continue

        for qt_label, sub in [
            ("all", per_query),
            ("movies", per_query[per_query["query_type"] == "movies"]),
            ("reviews", per_query[per_query["query_type"] == "reviews"]),
        ]:
            if sub.empty:
                continue
            t_oracle = sub["oracle_runtime"].sum()
            t_hnsw = sub["hnsw_runtime"].sum()
            t_ivf = sub["ivf_runtime"].sum()
            summary_rows.append(
                {
                    "system": system,
                    "query_type": qt_label,
                    "n_queries": len(sub),
                    "total_time_oracle_s": t_oracle,
                    "total_time_always_hnsw_s": t_hnsw,
                    "total_time_always_ivf_s": t_ivf,
                    "mean_qps_oracle": sub["oracle_qps"].mean(),
                    "mean_qps_always_hnsw": sub["hnsw_qps"].mean(),
                    "mean_qps_always_ivf": sub["ivf_qps"].mean(),
                    "speedup_vs_always_hnsw": t_hnsw / t_oracle if t_oracle > 0 else np.nan,
                    "speedup_vs_always_ivf": t_ivf / t_oracle if t_oracle > 0 else np.nan,
                    "oracle_hnsw_wins_pct": 100 * (sub["oracle_plan"] == hnsw_label).mean(),
                    "oracle_ivf_wins_pct": 100 * (sub["oracle_plan"] == ivf_label).mean(),
                }
            )

    return pd.DataFrame(summary_rows)


def plot_system_speedup(speedup_df: pd.DataFrame, output_path: Path, recall_target: float):
    """Bar chart of oracle speedup vs always-HNSW and always-IVF per system."""
    plot_df = speedup_df[speedup_df["query_type"] == "all"].copy()
    if plot_df.empty:
        return

    systems = plot_df["system"].tolist()
    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_h = ax.bar(x - width / 2, plot_df["speedup_vs_always_hnsw"], width, label="vs always HNSW", color="#3498db")
    bars_i = ax.bar(x + width / 2, plot_df["speedup_vs_always_ivf"], width, label="vs always IVF", color="#e67e22")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Speedup (total latency ratio)")
    ax.set_title(f"In-system oracle speedup (recall ≥ {recall_target}, tuned hyperparams)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bars in (bars_h, bars_i):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}×", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved speedup plot: {output_path}")


def verify_brute_force(df: pd.DataFrame) -> pd.DataFrame:
    bf = df[df["algorithm"].isin(BF_METHODS.keys())].copy()
    summary = (
        bf.groupby("algorithm")
        .agg(
            mean_recall=("recall", "mean"),
            min_recall=("recall", "min"),
            max_recall=("recall", "max"),
            n=("recall", "count"),
        )
        .reset_index()
    )
    summary["recall_ok"] = summary["min_recall"] >= 1.0 - 1e-9
    return summary


def pick_best_plan_per_query_for_system(
    df: pd.DataFrame,
    best_hp: pd.DataFrame,
    system_algos: list[str],
    recall_target: float = RECALL_TARGET,
) -> pd.DataFrame:
    """Pick best HNSW vs IVF vs BF plan within one system."""
    ann = _plans_at_tuned_hyperparams(df, best_hp, system_algos)
    if ann.empty:
        return pd.DataFrame()

    keys = ["query_id", "query_type", "filter_id"]
    records = []
    for key_vals, group in ann.groupby(keys):
        feasible = group[group["recall"] >= recall_target]
        if len(feasible) == 0:
            pick = group.sort_values(["recall", "qps"], ascending=[False, False]).iloc[0]
        else:
            pick = feasible.sort_values("qps", ascending=False).iloc[0]
        records.append(
            {
                "query_id": key_vals[0],
                "query_type": key_vals[1],
                "filter_id": key_vals[2],
                "best_algorithm": pick["algorithm"],
                "best_plan": plan_label(pick["algorithm"]),
                "best_recall": pick["recall"],
                "best_qps": pick["qps"],
                "filter_selectivity": pick["filter_selectivity"],
                "gls_correlation": pick["gls_correlation"],
            }
        )
    return pd.DataFrame(records)


def pick_best_plan_per_query(
    df: pd.DataFrame, best_hp: pd.DataFrame, recall_target: float = RECALL_TARGET
) -> pd.DataFrame:
    all_algos = list(ANN_METHODS.keys()) + list(BF_METHODS.keys())
    ann = _plans_at_tuned_hyperparams(df, best_hp, all_algos)
    if ann.empty:
        return pd.DataFrame()

    keys = ["query_id", "query_type", "filter_id"]
    records = []
    for key_vals, group in ann.groupby(keys):
        feasible = group[group["recall"] >= recall_target]
        if len(feasible) == 0:
            pick = group.sort_values(["recall", "qps"], ascending=[False, False]).iloc[0]
        else:
            pick = feasible.sort_values("qps", ascending=False).iloc[0]
        records.append(
            {
                "query_id": key_vals[0],
                "query_type": key_vals[1],
                "filter_id": key_vals[2],
                "best_algorithm": pick["algorithm"],
                "best_plan": plan_label(pick["algorithm"]),
                "best_recall": pick["recall"],
                "best_qps": pick["qps"],
                "filter_selectivity": pick["filter_selectivity"],
                "gls_correlation": pick["gls_correlation"],
            }
        )
    return pd.DataFrame(records)


def plot_best_plan_scatter(
    best_plans: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
    plan_order: list[str] | None = None,
):
    plot_df = best_plans.dropna(subset=["filter_selectivity", "gls_correlation"]).copy()
    if plot_df.empty:
        print("No data for scatter plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    plans = plan_order if plan_order else sorted(plot_df["best_plan"].unique())
    for plan in plans:
        sub = plot_df[plot_df["best_plan"] == plan]
        if sub.empty:
            continue
        ax.scatter(
            sub["filter_selectivity"],
            sub["gls_correlation"],
            label=plan,
            alpha=0.55,
            s=35,
            c=PLAN_COLORS.get(plan, "gray"),
            edgecolors="none",
        )

    ax.set_xlabel("Filter selectivity")
    ax.set_ylabel("GLS correlation")
    ax.set_title(f"Best plan per query (recall ≥ {RECALL_TARGET}){title_suffix}")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.legend(title="Best plan", loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter: {output_path}")


def plot_best_plan_by_system(df: pd.DataFrame, best_hp: pd.DataFrame, output_dir: Path, recall_target: float):
    """One scatter per system × query type (4 plots: FAISS/pgvector × movies/reviews)."""
    for system, algos in SYSTEMS.items():
        system_plans = pick_best_plan_per_query_for_system(df, best_hp, algos, recall_target)
        plan_labels = [plan_label(a) for a in algos]
        for qt in ["movies", "reviews"]:
            sub = system_plans[system_plans["query_type"] == qt]
            plot_best_plan_scatter(
                sub,
                output_dir / f"best_plan_scatter_{system.lower()}_{qt}.png",
                title_suffix=f" — {system}, {qt}",
                plan_order=plan_labels,
            )


def plot_plan_distribution(best_plans: pd.DataFrame, output_path: Path):
    """Stacked bar: fraction of best plans by selectivity bin."""
    df = best_plans.dropna(subset=["filter_selectivity"]).copy()
    df["sel_bin"] = pd.cut(
        df["filter_selectivity"],
        bins=[0, 0.01, 0.05, 0.1, 0.2, 1.0],
        labels=["<1%", "1-5%", "5-10%", "10-20%", ">20%"],
    )
    ct = pd.crosstab(df["sel_bin"], df["best_plan"], normalize="index")
    ax = ct.plot(kind="bar", stacked=True, figsize=(10, 6), color=[PLAN_COLORS.get(c, "gray") for c in ct.columns])
    ax.set_xlabel("Selectivity bin")
    ax.set_ylabel("Fraction of queries")
    ax.set_title("Best plan distribution by selectivity")
    ax.legend(title="Best plan", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plan distribution: {output_path}")


def plot_qps_recall_frontier(df: pd.DataFrame, best_hp: pd.DataFrame, output_path: Path):
    """QPS-recall curves for each method using all hyperparams."""
    # Colors for this plot only — FAISS vs pgvector must be distinguishable.
    frontier_colors = {
        "FAISS HNSW pre": "#2ecc71",
        "FAISS HNSW post": "#27ae60",
        "FAISS IVF pre": "#1a5276",
        "FAISS IVF post": "#3498db",
        "pgvector HNSW": "#4488FF",
        "pgvector IVF": "#9b59b6",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, idx_type in zip(axes, ["HNSW", "IVF"]):
        methods = [a for a, m in ANN_METHODS.items() if m["index"] == idx_type]
        for algo in methods:
            sub = df[df["algorithm"] == algo]
            curve = (
                sub.groupby("hyperparam")
                .agg(mean_recall=("recall", "mean"), mean_qps=("qps", "mean"))
                .reset_index()
                .sort_values("mean_recall")
            )
            label = ANN_METHODS[algo]["label"]
            color = frontier_colors[label]
            ax.plot(curve["mean_recall"], curve["mean_qps"], "o-", label=label, color=color)
            best = best_hp[best_hp["algorithm"] == algo].iloc[0]
            ax.scatter([best["mean_recall"]], [best["mean_qps"]], s=120, marker="*", color=color, zorder=5)
        ax.axvline(RECALL_TARGET, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("Mean recall")
        ax.set_ylabel("Mean QPS")
        ax.set_yscale("log")
        ax.set_title(idx_type)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("QPS vs recall (★ = chosen hyperparameter)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved QPS-recall frontier: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Query optimizer analysis")
    parser.add_argument(
        "--results-dir",
        default="results/MoRe_UPD_large_attidx_0",
        help="Benchmark results directory",
    )
    parser.add_argument("--dataset-size", default="large")
    parser.add_argument(
        "--filter-stats",
        default="data/datasets/MoRe_large/stats/filter_stats_0_k2048.csv",
    )
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    parser.add_argument("--output-dir", default="analysis/plots/query_optimizer")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir
    root_data = root / "data" / "datasets"
    filter_stats = root / args.filter_stats
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building results dataframe from HDF5...")
    df = build_results_df(results_dir, root_data, args.dataset_size)
    if df.empty:
        print("No results found!")
        return

    print(f"  {len(df)} rows, {df['algorithm'].nunique()} algorithms")
    df = attach_gls_correlation(df, filter_stats, root_data, args.dataset_size)

    csv_path = output_dir / "all_query_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    print("\n=== Brute-force recall check ===")
    bf_summary = verify_brute_force(df)
    print(bf_summary.to_string(index=False))
    bf_summary.to_csv(output_dir / "brute_force_recall_check.csv", index=False)

    print(f"\n=== Best hyperparameters (mean recall >= {args.recall_target}) ===")
    best_hp = find_best_hyperparams(df, args.recall_target)
    print(best_hp.to_string(index=False))
    best_hp.to_csv(output_dir / "best_hyperparameters.csv", index=False)

    print(f"\n=== Hyperparameter sweep & recall≥{args.recall_target} recommendations ===")
    hp_sweep, hp_recs = analyze_hyperparam_recommendations(df, args.recall_target)
    hp_sweep.to_csv(output_dir / "hyperparam_sweep.csv", index=False)
    hp_recs.to_csv(output_dir / "hyperparam_recommendations.csv", index=False)
    print(hp_sweep.to_string(index=False))
    print("\nRecommendations:")
    for _, row in hp_recs.iterrows():
        print(f"  {row['label']}: {row['recommended_eval_hyperparams']}")

    print(f"\n=== In-system oracle speedup (best HNSW/IVF per query, recall ≥ {args.recall_target}) ===")
    speedup_df = compute_system_oracle_speedup(df, best_hp, args.recall_target)
    speedup_df.to_csv(output_dir / "system_oracle_speedup.csv", index=False)
    print(speedup_df.to_string(index=False))
    plot_system_speedup(speedup_df, output_dir / "system_oracle_speedup.png", args.recall_target)

    print("\n=== Best plan per query (global, all methods) ===")
    best_plans = pick_best_plan_per_query(df, best_hp, args.recall_target)
    best_plans.to_csv(output_dir / "best_plan_per_query.csv", index=False)
    print(f"  {len(best_plans)} queries")
    print(best_plans["best_plan"].value_counts().to_string())

    print("\n=== Generating plots ===")
    plot_best_plan_scatter(
        best_plans,
        output_dir / "best_plan_scatter_all.png",
        plan_order=list(PLAN_COLORS.keys()),
    )
    plot_best_plan_by_system(df, best_hp, output_dir, args.recall_target)
    plot_plan_distribution(best_plans, output_dir / "best_plan_by_selectivity_bin.png")
    plot_qps_recall_frontier(df, best_hp, output_dir / "qps_recall_frontier.png")

    # Confusion-style: which plans win in GLS correlation tertiles
    bp = best_plans.dropna(subset=["gls_correlation"]).copy()
    bp["gls_tertile"] = pd.qcut(bp["gls_correlation"], 3, labels=["low", "mid", "high"])
    tertile_ct = pd.crosstab(bp["gls_tertile"], bp["best_plan"])
    tertile_ct.to_csv(output_dir / "best_plan_by_gls_tertile.csv")
    print("\nBest plan counts by GLS tertile:")
    print(tertile_ct.to_string())


if __name__ == "__main__":
    main()
