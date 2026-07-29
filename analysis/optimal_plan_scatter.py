#!/usr/bin/env python3
"""
Optimal-plan scatter plots for filtered ANN benchmarks.

For each (query, filter, k), the best-plan latency t_b is the minimum runtime
among all candidate (algorithm, hyperparam) plans that meet the recall target.
A specific (index type, HP) combination is optimal for that workload if:

    recall >= recall_target  and  runtime <= t_b * (1 + eps)

Default eps = 0.1.

Produces one figure per index type (algorithm), with one subplot per swept
hyperparameter. When multiple top-k values are present (default: all of
{10,20,40}), each (query, filter, algo, HP) is plotted once and colored by how
many k values mark it optimal:

    0 -> grey, 1 -> light green, 2 -> green, 3 -> dark green

Separate plot trees for exact and estimated GLS correlation. Duplicate
benchmark runs for the same (query, algo, hp, k) are averaged.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_optimizer_analysis import (
    FAISS_ALL_ALGOS,
    RECALL_TARGET,
    SYSTEMS,
    attach_gls_correlation,
    build_results_df,
    plan_label,
    _present_algos,
)

EPS_DEFAULT = 0.1

# Color by # of top-k values for which (query, filter, algo, hp) is optimal.
NUM_OPTIMAL_COLORS = {
    0: "#d5d8dc",  # grey
    1: "#a9dfbf",  # light green
    2: "#27ae60",  # green
    3: "#145a32",  # dark green
}
NUM_OPTIMAL_LABELS = {
    0: "0 k optimal",
    1: "1 k optimal",
    2: "2 k optimal",
    3: "3 k optimal",
}

# Workload identity for t_b (includes k when present).
QUERY_KEYS = ["query_id", "query_type", "filter_id"]
# Point identity after collapsing across k for the scatter.
POINT_KEYS = ["query_id", "query_type", "filter_id", "algorithm", "hyperparam"]


def _candidate_algo_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Named candidate pools used to define t_b and the index types to plot."""
    groups: dict[str, list[str]] = {}
    faiss_all = _present_algos(df, FAISS_ALL_ALGOS)
    if faiss_all:
        groups["faiss_all"] = faiss_all
    for system, algos in SYSTEMS.items():
        present = _present_algos(df, algos)
        if present:
            groups[system.lower()] = present
    return groups


def _workload_keys(df: pd.DataFrame) -> list[str]:
    """Keys that identify one decision unit: (query, filter[, k])."""
    keys = list(QUERY_KEYS)
    if "k" in df.columns:
        keys.append("k")
    return keys


def prepare_plans_df(df: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """
    Optionally restrict to one top-k and collapse duplicate benchmark runs.

    t_b is computed per (query, filter, k). Pass k=None to keep all top-k values.
    """
    out = df.copy()
    if k is not None and "k" in out.columns:
        out = out[out["k"] == k]
        if out.empty:
            return out
    group_keys = _workload_keys(out) + ["algorithm", "hyperparam"]

    agg: dict[str, str] = {
        "recall": "mean",
        "runtime": "mean",
        "filter_selectivity": "first",
        "gls_correlation": "first",
        "hyperparam_name": "first",
    }
    if "qps" in out.columns:
        agg["qps"] = "mean"
    present = {c: f for c, f in agg.items() if c in out.columns}
    return out.groupby(group_keys, as_index=False).agg(present)


def compute_best_latency_per_query(
    df: pd.DataFrame,
    algos: list[str],
    recall_target: float = RECALL_TARGET,
) -> pd.DataFrame:
    """
    Per-(query, filter, k) best latency t_b over swept (algorithm, hyperparam).

    Only plans with recall >= recall_target compete; if none qualify, fall back to
    the highest-recall plan (same rule as best-plan oracle).
    """
    plans = df[df["algorithm"].isin(algos)].copy()
    if plans.empty:
        return pd.DataFrame()

    keys = _workload_keys(plans)
    feasible = plans[plans["recall"] >= recall_target]
    feas_pick = (
        feasible.sort_values("runtime", kind="mergesort")
        .groupby(keys, sort=False)
        .head(1)
        if not feasible.empty
        else pd.DataFrame(columns=plans.columns)
    )

    all_keys = plans[keys].drop_duplicates()
    if len(feas_pick):
        uncovered = all_keys.merge(feas_pick[keys], on=keys, how="left", indicator=True)
        uncovered = uncovered[uncovered["_merge"] == "left_only"][keys]
    else:
        uncovered = all_keys

    if len(uncovered):
        fallback_pool = plans.merge(uncovered, on=keys, how="inner")
        fallback = (
            fallback_pool.sort_values(
                ["recall", "runtime"], ascending=[False, True], kind="mergesort"
            )
            .groupby(keys, sort=False)
            .head(1)
        )
        pick = (
            pd.concat([feas_pick, fallback], ignore_index=True)
            if len(feas_pick)
            else fallback
        )
    else:
        pick = feas_pick

    pick = pick.rename(
        columns={
            "runtime": "t_b",
            "algorithm": "best_algorithm",
            "hyperparam": "best_hyperparam",
            "hyperparam_name": "best_hyperparam_name",
            "recall": "best_recall",
        }
    )
    pick["best_plan"] = pick["best_algorithm"].map(plan_label)
    pick["best_hyperparam"] = pick["best_hyperparam"].astype(int)
    keep = keys + [
        "t_b",
        "best_algorithm",
        "best_plan",
        "best_hyperparam",
        "best_hyperparam_name",
        "best_recall",
        "filter_selectivity",
        "gls_correlation",
    ]
    return pick[keep].reset_index(drop=True)


def mark_optimal_plans(
    df: pd.DataFrame,
    best_latency: pd.DataFrame,
    algos: list[str],
    eps: float,
    recall_target: float = RECALL_TARGET,
) -> pd.DataFrame:
    """Attach t_b and is_optimal for every (query, filter, k, algorithm, hyperparam)."""
    plans = df[df["algorithm"].isin(algos)].copy()
    if plans.empty or best_latency.empty:
        return pd.DataFrame()

    keys = _workload_keys(plans)
    merge_cols = keys + [
        "t_b",
        "best_algorithm",
        "best_plan",
        "best_hyperparam",
        "best_hyperparam_name",
        "best_recall",
    ]
    merged = plans.merge(
        best_latency[merge_cols],
        on=keys,
        how="inner",
    )
    threshold = merged["t_b"] * (1.0 + eps)
    merged["latency_threshold"] = threshold
    merged["is_optimal"] = (merged["recall"] >= recall_target) & (
        merged["runtime"] <= threshold
    )
    merged["plan_label"] = merged["algorithm"].map(plan_label)
    return merged


def aggregate_optimal_across_k(optimal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-k optimality into one point per (query, filter, algo, hp).

    num_optimal_k = number of distinct k values where is_optimal is True.
    Selectivity / GLS are taken from the first row (identical across k).
    """
    if optimal_df.empty:
        return optimal_df

    has_k = "k" in optimal_df.columns
    group_cols = list(POINT_KEYS)
    agg: dict[str, str] = {
        "is_optimal": "sum",
        "filter_selectivity": "first",
        "gls_correlation": "first",
        "hyperparam_name": "first",
        "plan_label": "first",
    }
    if has_k:
        agg["k"] = "nunique"

    grouped = optimal_df.groupby(group_cols, as_index=False).agg(agg)
    grouped = grouped.rename(
        columns={
            "is_optimal": "num_optimal_k",
            "k": "n_k_present",
        }
    )
    if not has_k:
        grouped["n_k_present"] = 1
        grouped["num_optimal_k"] = grouped["num_optimal_k"].astype(int)
    else:
        grouped["num_optimal_k"] = grouped["num_optimal_k"].astype(int)
        grouped["n_k_present"] = grouped["n_k_present"].astype(int)

    # Cap color index at 3 (k in {10,20,40}).
    grouped["num_optimal_k"] = grouped["num_optimal_k"].clip(upper=3)
    return grouped


def _subplot_grid(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    ncols = min(5, n)
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def plot_optimal_scatter_for_algorithm(
    optimal_df: pd.DataFrame,
    algorithm: str,
    output_path: Path,
    eps: float,
    recall_target: float,
    gls_source: str,
    query_type: str | None = None,
    title_suffix: str = "",
    max_k_count: int = 3,
):
    """One figure for an index type; each subplot is one hyperparameter value.

    Points are colored by num_optimal_k (how many top-k values mark the plan
    optimal for that query/filter).
    """
    sub = optimal_df[optimal_df["algorithm"] == algorithm].copy()
    if query_type is not None:
        sub = sub[sub["query_type"] == query_type]
    sub = sub.dropna(subset=["filter_selectivity", "gls_correlation"])
    if sub.empty:
        print(f"  skip {algorithm}{title_suffix}: no data")
        return

    if "num_optimal_k" not in sub.columns:
        # Per-k rows: treat boolean is_optimal as 0/1 count.
        sub["num_optimal_k"] = sub["is_optimal"].astype(int)

    label = plan_label(algorithm)
    hp_name = sub["hyperparam_name"].iloc[0]
    hps = sorted(sub["hyperparam"].unique())
    nrows, ncols = _subplot_grid(len(hps))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.8 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    if gls_source == "estimated":
        gls_label = "GLS correlation (estimated)"
        gls_note = "estimated GLS"
    else:
        gls_label = "GLS correlation (exact k=2048)"
        gls_note = "exact GLS"

    color_levels = list(range(0, max_k_count + 1))

    for i, hp in enumerate(hps):
        ax = axes[i // ncols][i % ncols]
        hp_df = sub[sub["hyperparam"] == hp]
        # Draw low counts first so darker greens stay on top.
        for n_opt in color_levels:
            pts = hp_df[hp_df["num_optimal_k"] == n_opt]
            if pts.empty:
                continue
            ax.scatter(
                pts["filter_selectivity"],
                pts["gls_correlation"],
                c=NUM_OPTIMAL_COLORS.get(n_opt, "#d5d8dc"),
                s=10 if n_opt == 0 else 14 + 2 * n_opt,
                alpha=0.35 if n_opt == 0 else 0.75,
                edgecolors="none",
                zorder=1 + n_opt,
                label=NUM_OPTIMAL_LABELS.get(n_opt, f"{n_opt} k optimal"),
            )
        n_any = int((hp_df["num_optimal_k"] > 0).sum())
        n_tot = len(hp_df)
        pct = 100.0 * n_any / n_tot if n_tot else 0.0
        # Breakdown of counts among points with at least one optimal k.
        parts = []
        for n_opt in color_levels[1:]:
            c = int((hp_df["num_optimal_k"] == n_opt).sum())
            if c:
                parts.append(f"{n_opt}k:{c}")
        breakdown = (", ".join(parts)) if parts else "none"
        if hp_name == "none":
            ax.set_title(
                f"BF\n{pct:.0f}% any-opt ({n_any}/{n_tot})\n{breakdown}",
                fontsize=8,
            )
        else:
            ax.set_title(
                f"{hp_name}={int(hp)}\n{pct:.0f}% any-opt ({n_any}/{n_tot})\n{breakdown}",
                fontsize=8,
            )
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.35)
        ax.grid(True, alpha=0.25)

    for j in range(len(hps), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    for ax in axes[-1]:
        ax.set_xlabel("Filter selectivity", fontsize=9)
    for ax_row in axes:
        ax_row[0].set_ylabel(gls_label, fontsize=9)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NUM_OPTIMAL_COLORS[n],
            markersize=8,
            label=NUM_OPTIMAL_LABELS[n],
        )
        for n in color_levels
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        title="# k optimal",
    )
    fig.suptitle(
        f"Optimal plans: {label} (recall ≥ {recall_target}, "
        f"runtime ≤ t_b·(1+{eps:g}), {gls_note}){title_suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def summarize_optimal_rates(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Per (algorithm, hyperparam, query_type) rates by num_optimal_k."""
    group_cols = ["algorithm", "plan_label", "hyperparam", "hyperparam_name", "query_type"]
    rows = []
    for keys, g in plot_df.groupby(group_cols, dropna=False):
        algo, label, hp, hp_name, qt = keys
        n = len(g)
        row = {
            "algorithm": algo,
            "plan_label": label,
            "hyperparam": int(hp),
            "hyperparam_name": hp_name,
            "query_type": qt,
            "n_queries": n,
            "n_any_optimal": int((g["num_optimal_k"] > 0).sum()),
            "any_optimal_pct": 100.0 * (g["num_optimal_k"] > 0).mean() if n else np.nan,
            "mean_num_optimal_k": float(g["num_optimal_k"].mean()) if n else np.nan,
        }
        for n_opt in (0, 1, 2, 3):
            row[f"n_opt_k_{n_opt}"] = int((g["num_optimal_k"] == n_opt).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["algorithm", "query_type", "hyperparam"])


def generate_optimal_plots(
    df: pd.DataFrame,
    output_dir: Path,
    eps: float,
    recall_target: float,
    gls_source: str,
    groups: dict[str, list[str]] | None = None,
    write_flags: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = groups or _candidate_algo_groups(df)
    n_k = int(df["k"].nunique()) if "k" in df.columns else 1
    max_k_count = min(3, n_k)

    for group_name, algos in groups.items():
        print(f"\n=== Optimal-plan plots: {group_name} ({gls_source} GLS, {n_k} k values) ===")
        best_lat = compute_best_latency_per_query(df, algos, recall_target)
        if best_lat.empty:
            print(f"  no best-latency rows for {group_name}")
            continue

        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        best_lat.to_csv(group_dir / "best_latency_per_query.csv", index=False)

        optimal_df = mark_optimal_plans(df, best_lat, algos, eps, recall_target)
        plot_df = aggregate_optimal_across_k(optimal_df)

        summary = summarize_optimal_rates(plot_df)
        summary.to_csv(group_dir / "optimal_plan_rates.csv", index=False)
        print(summary.to_string(index=False))

        if write_flags:
            flag_cols = [
                c
                for c in [
                    "query_id",
                    "query_type",
                    "filter_id",
                    "k",
                    "algorithm",
                    "hyperparam",
                    "hyperparam_name",
                    "recall",
                    "runtime",
                    "t_b",
                    "latency_threshold",
                    "is_optimal",
                    "filter_selectivity",
                    "gls_correlation",
                ]
                if c in optimal_df.columns
            ]
            optimal_df[flag_cols].to_csv(
                group_dir / "optimal_flags_per_query_hp.csv", index=False
            )
            plot_df.to_csv(group_dir / "optimal_counts_across_k.csv", index=False)

        k_note = f" — color = #k optimal (of {n_k})"
        for algo in algos:
            safe = algo.replace("(", "_").replace(")", "").replace("-", "_")
            plot_optimal_scatter_for_algorithm(
                plot_df,
                algo,
                group_dir / f"optimal_scatter_{safe}_all.png",
                eps=eps,
                recall_target=recall_target,
                gls_source=gls_source,
                query_type=None,
                title_suffix=f" — all queries{k_note}",
                max_k_count=max_k_count,
            )
            for qt in ["movies", "reviews"]:
                plot_optimal_scatter_for_algorithm(
                    plot_df,
                    algo,
                    group_dir / f"optimal_scatter_{safe}_{qt}.png",
                    eps=eps,
                    recall_target=recall_target,
                    gls_source=gls_source,
                    query_type=qt,
                    title_suffix=f" — {qt}{k_note}",
                    max_k_count=max_k_count,
                )


def load_or_build_df(
    results_csv: Path | None,
    results_dir: Path,
    root_data: Path,
    dataset_size: str,
    gls_stats: Path,
) -> pd.DataFrame:
    if results_csv is not None and results_csv.is_file():
        print(f"Loading results from {results_csv}")
        df = pd.read_csv(results_csv)
        if "gls_correlation" not in df.columns:
            print(f"  attaching GLS from {gls_stats}")
            df = attach_gls_correlation(df, gls_stats, root_data, dataset_size)
        return df

    print(f"Building results from HDF5 under {results_dir}...")
    df_base = build_results_df(results_dir, root_data, dataset_size)
    if df_base.empty:
        return df_base
    return attach_gls_correlation(df_base, gls_stats, root_data, dataset_size)


def main():
    parser = argparse.ArgumentParser(description="Optimal-plan scatter plots")
    parser.add_argument(
        "--results-dir",
        default="results/MoRe_UPD_large_attidx_0",
        help="Benchmark results directory (used if --results-csv is absent)",
    )
    parser.add_argument(
        "--results-csv",
        default="analysis/plots/query_optimizer/all_query_results.csv",
        help="Prebuilt per-query results CSV (skips HDF5 rebuild when present)",
    )
    parser.add_argument(
        "--gls-est-results-csv",
        default="analysis/plots/query_optimizer/gls_est/all_query_results.csv",
        help="Prebuilt results CSV with estimated GLS correlation",
    )
    parser.add_argument("--dataset-size", default="large")
    parser.add_argument(
        "--filter-stats",
        default="data/datasets/MoRe_large/stats/filter_stats_0_k2048.csv",
        help="Exact GLS correlation stats",
    )
    parser.add_argument(
        "--gls-est-stats",
        default="data/datasets/MoRe_large/stats/gls_correlation_estimates_ivf_lsek2048_a1_0.csv",
        help="Estimated GLS correlation stats",
    )
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Top-k to analyze. Default 0 = all k values (color by #k optimal). "
        "Pass 10/20/40 for a single k.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=EPS_DEFAULT,
        help="Optimality slack: runtime <= t_b * (1 + eps)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/plots/query_optimizer/optimal",
        help="Output directory for exact-GLS optimal plots",
    )
    parser.add_argument(
        "--gls-est-output-dir",
        default="analysis/plots/query_optimizer/optimal/gls_est",
        help="Output directory for estimated-GLS optimal plots",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional subset of candidate groups (faiss_all, faiss, pgvector)",
    )
    parser.add_argument(
        "--write-flags",
        action="store_true",
        help="Also write per-(query, algo, hp) optimal_flags_per_query_hp.csv (large)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir
    root_data = root / "data" / "datasets"
    exact_gls_stats = root / args.filter_stats
    gls_est_stats = root / args.gls_est_stats
    results_csv = root / args.results_csv if args.results_csv else None
    gls_est_csv = root / args.gls_est_results_csv if args.gls_est_results_csv else None
    output_dir = root / args.output_dir
    gls_est_output_dir = root / args.gls_est_output_dir

    # --- exact GLS ---
    df_exact = load_or_build_df(
        results_csv,
        results_dir,
        root_data,
        args.dataset_size,
        exact_gls_stats,
    )
    if df_exact.empty:
        print("No results found!")
        return

    k_filter = None if args.k == 0 else args.k
    print(f"Preparing plans (k={k_filter if k_filter is not None else 'all'}, dedupe runs)...")
    n_before = len(df_exact)
    df_exact = prepare_plans_df(df_exact, k=k_filter)
    print(f"  {n_before} -> {len(df_exact)} rows")
    if df_exact.empty:
        print(f"No rows left after filtering k={k_filter}")
        return

    groups = _candidate_algo_groups(df_exact)
    if args.groups:
        groups = {k: v for k, v in groups.items() if k in args.groups}
        if not groups:
            print(f"No matching groups in {args.groups}; available: {list(_candidate_algo_groups(df_exact))}")
            return

    print(f"Exact GLS: {len(df_exact)} rows, groups={list(groups)}")
    generate_optimal_plots(
        df_exact,
        output_dir,
        eps=args.eps,
        recall_target=args.recall_target,
        gls_source="exact",
        groups=groups,
        write_flags=args.write_flags,
    )

    # --- estimated GLS ---
    if gls_est_csv is not None and gls_est_csv.is_file():
        print(f"\nLoading estimated-GLS results from {gls_est_csv}")
        df_est = pd.read_csv(gls_est_csv)
    elif gls_est_stats.is_file():
        print(f"\nAttaching estimated GLS from {gls_est_stats}")
        drop_cols = [c for c in ("gls_correlation", "estimator") if c in df_exact.columns]
        df_est = attach_gls_correlation(
            df_exact.drop(columns=drop_cols),
            gls_est_stats,
            root_data,
            args.dataset_size,
        )
    else:
        print(f"\nSkipping estimated-GLS plots: neither {gls_est_csv} nor {gls_est_stats} found")
        return

    if "gls_correlation" not in df_est.columns:
        print("Estimated results missing gls_correlation; skipping")
        return

    n_before = len(df_est)
    df_est = prepare_plans_df(df_est, k=k_filter)
    print(f"  estimated GLS prepare: {n_before} -> {len(df_est)} rows")
    if df_est.empty:
        print(f"No estimated-GLS rows left after filtering k={k_filter}")
        return

    est_groups = {k: v for k, v in _candidate_algo_groups(df_est).items() if k in groups}
    print(f"Estimated GLS: {len(df_est)} rows, groups={list(est_groups)}")
    generate_optimal_plots(
        df_est,
        gls_est_output_dir,
        eps=args.eps,
        recall_target=args.recall_target,
        gls_source="estimated",
        groups=est_groups,
        write_flags=args.write_flags,
    )


if __name__ == "__main__":
    main()
