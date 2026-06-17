#!/usr/bin/env python3
"""
Summarize query latencies, recall, and QPS by selectivity.

- Sums all query latencies across all queries (overall total)
- Sums query latencies and computes recall at each selectivity label
- Prints a table with recall, QPS, QPS increase (Nx), and recall increase (diff)
- Compares att_idx=0 (baseline) vs att_idx=1 (with attribute index)

Usage:
  python summarize_latency_recall_by_selectivity.py --dataset_size large
  python summarize_latency_recall_by_selectivity.py --index_type hnsw --query_type movies
"""

import argparse
import os
import pandas as pd


def add_query_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add query_type column from query_id (qm* -> movies, qr* -> reviews)."""
    df = df.copy()
    df['query_type'] = df['query_id'].apply(lambda x: 'movies' if str(x).startswith('qm') else 'reviews')
    return df


def compute_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Compute sum of latencies, mean recall, n_queries, and QPS for each group.

    QPS = n_queries / sum(runtime)  (queries per second when run sequentially)
    """
    agg = df.groupby(group_cols).agg(
        sum_latency_s=('runtime', 'sum'),
        mean_recall=('recall', 'mean'),
        n_queries=('query_id', 'count'),
    ).reset_index()
    agg['QPS'] = agg['n_queries'] / agg['sum_latency_s']
    return agg


def compute_comparison(base: pd.DataFrame, comp: pd.DataFrame, merge_cols: list[str]) -> pd.DataFrame:
    """
    Merge base and comp on merge_cols, compute QPS increase (Nx) and recall increase (diff).
    """
    base = base.rename(columns={'QPS': 'QPS_base', 'mean_recall': 'recall_base', 'sum_latency_s': 'sum_latency_base'})
    comp = comp.rename(columns={'QPS': 'QPS_comp', 'mean_recall': 'recall_comp', 'sum_latency_s': 'sum_latency_comp'})
    merged = base[merge_cols + ['QPS_base', 'recall_base']].merge(
        comp[merge_cols + ['QPS_comp', 'recall_comp']],
        on=merge_cols,
        how='outer',
    )
    merged['QPS_increase_Nx'] = merged['QPS_comp'] / merged['QPS_base'].replace(0, float('nan'))
    merged['recall_increase'] = merged['recall_comp'] - merged['recall_base']
    return merged


def print_table(summary: pd.DataFrame, title: str):
    """Print a formatted table with QPS increase (Nx) and recall increase (diff)."""
    print(f"\n{title}")
    print("=" * 95)

    display = summary.copy()
    if 'QPS_base' in display.columns:
        display['QPS_base'] = display['QPS_base'].round(1)
    if 'QPS_comp' in display.columns:
        display['QPS_comp'] = display['QPS_comp'].round(1)
    if 'QPS_increase_Nx' in display.columns:
        display['QPS_increase_Nx'] = display['QPS_increase_Nx'].apply(
            lambda x: f"{x:.2f}x" if pd.notna(x) and x != float('inf') else "—"
        )
    if 'recall_base' in display.columns:
        display['recall_base'] = display['recall_base'].round(4)
    if 'recall_comp' in display.columns:
        display['recall_comp'] = display['recall_comp'].round(4)
    if 'recall_increase' in display.columns:
        display['recall_increase'] = display['recall_increase'].apply(
            lambda x: f"{x:+.4f}" if pd.notna(x) else "—"
        )

    # Column order for display
    desired = ['query_type', 'filter_selectivity', 'recall_base', 'recall_comp', 'recall_increase',
               'QPS_base', 'QPS_comp', 'QPS_increase_Nx']
    cols = [c for c in desired if c in display.columns]
    display = display[[c for c in cols] + [c for c in display.columns if c not in cols]]

    print(display.to_string(index=False))
    print("=" * 95)


def run_summary(
    root_results: str,
    dataset_size: str,
    index_type: str,
    query_type: str | None,
    algorithm: str | None,
    k: int | None,
):
    """
    Load data for att_idx=0 and att_idx=1, compute summaries, compare, and print tables.
    """
    att_base, att_comp = 0, 1
    results = []

    for idx_name, csv_suffix in [('HNSW', 'hnsw'), ('IVF', 'ivf')]:
        if index_type not in (idx_name.lower(), 'both'):
            continue

        path_base = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_{att_base}/all_results_{csv_suffix}.csv"
        path_comp = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_{att_comp}/all_results_{csv_suffix}.csv"

        if not os.path.exists(path_base):
            print(f"Warning: Baseline CSV not found at {path_base}")
            continue
        if not os.path.exists(path_comp):
            print(f"Warning: Compare CSV not found at {path_comp}")
            continue

        def load_and_filter(path: str) -> pd.DataFrame:
            df = pd.read_csv(path)
            df = add_query_type(df)
            if algorithm:
                df = df[df['algorithm'] == algorithm]
            if k is not None:
                df = df[df['k'] == k]
            if query_type:
                df = df[df['query_type'] == query_type]
            return df

        df_base = load_and_filter(path_base)
        df_comp = load_and_filter(path_comp)

        if len(df_base) == 0 or len(df_comp) == 0:
            print(f"\n{idx_name}: No data after filters")
            continue

        results.append((idx_name, df_base, df_comp))

    for idx_name, df_base, df_comp in results:
        group_cols = ['filter_selectivity']
        if query_type is None:
            group_cols = ['query_type', 'filter_selectivity']

        summary_base = compute_summary(df_base, group_cols).sort_values(group_cols)
        summary_comp = compute_summary(df_comp, group_cols).sort_values(group_cols)

        # Overall totals
        total_lat_base = df_base['runtime'].sum()
        total_lat_comp = df_comp['runtime'].sum()
        n_base = len(df_base)
        n_comp = len(df_comp)
        qps_base = n_base / total_lat_base if total_lat_base > 0 else 0
        qps_comp = n_comp / total_lat_comp if total_lat_comp > 0 else 0
        recall_base = df_base['recall'].mean()
        recall_comp = df_comp['recall'].mean()
        qps_increase = qps_comp / qps_base if qps_base > 0 else float('nan')
        recall_increase = recall_comp - recall_base

        print(f"\n{'#' * 90}")
        print(f"# {idx_name} — att_idx=0 (baseline) vs att_idx=1 (with attr index), dataset_size={dataset_size}"
              + (f", query_type={query_type}" if query_type else "")
              + (f", algorithm={algorithm}" if algorithm else "")
              + (f", k={k}" if k else ""))
        print(f"{'#' * 90}")

        print(f"\n--- OVERALL (all selectivities) ---")
        print(f"  att_idx=0: sum_latency={total_lat_base:.2f}s, recall={recall_base:.4f}, QPS={qps_base:.1f}")
        print(f"  att_idx=1: sum_latency={total_lat_comp:.2f}s, recall={recall_comp:.4f}, QPS={qps_comp:.1f}")
        print(f"  QPS increase: {qps_increase:.2f}x" if pd.notna(qps_increase) else "  QPS increase: —")
        print(f"  Recall increase: {recall_increase:+.4f}")

        # Per selectivity comparison
        merged = compute_comparison(summary_base, summary_comp, group_cols)
        merged = merged.sort_values(group_cols)

        title = f"--- Per selectivity ({idx_name}): att_idx=0 vs att_idx=1 ---"
        print_table(merged, title)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(script_dir, '..', 'results_prev'))

    parser = argparse.ArgumentParser(
        description="Summarize query latencies, recall, and QPS by selectivity."
    )
    parser.add_argument(
        "--root_results",
        default=default_root,
        help="Root directory for results (default: results_prev)",
    )
    parser.add_argument(
        "--dataset_size",
        default="large",
        choices=["small", "medium", "large"],
        help="Dataset size",
    )
    parser.add_argument(
        "--index_type",
        default="both",
        choices=["hnsw", "ivf", "both"],
        help="Index type to analyze",
    )
    parser.add_argument(
        "--query_type",
        default=None,
        choices=["movies", "reviews"],
        help="Filter to specific query type (default: both)",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Filter to specific algorithm (e.g. pgvector, pgvector_ivf)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Filter to specific k (default: all)",
    )

    args = parser.parse_args()

    run_summary(
        root_results=args.root_results,
        dataset_size=args.dataset_size,
        index_type=args.index_type,
        query_type=args.query_type,
        algorithm=args.algorithm,
        k=args.k,
    )


if __name__ == "__main__":
    main()
