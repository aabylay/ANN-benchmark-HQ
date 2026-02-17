import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Set matplotlib to use DejaVu Serif (LaTeX-like font) and configure mathtext
mpl.rcParams['lines.linewidth'] = 2
plt.rc('font', family='serif', serif='DejaVu Serif', size=24)
plt.rc('mathtext', default='regular')

ROOT_RESULTS = '/home/abylay/ann-benchmarks-HQ/results'

# Paths — HNSW
DEFAULT_HNSW_CSV = f"{ROOT_RESULTS}/MoRe_UPD_large_attidx_0/all_results_hnsw.csv"
SEG16GB_HNSW_CSV = f"{ROOT_RESULTS}/ablation_seg16384/MoRe_UPD_large_attidx_0/all_results_hnsw.csv"
# Paths — IVF
DEFAULT_IVF_CSV  = f"{ROOT_RESULTS}/MoRe_UPD_large_attidx_0/all_results_ivf.csv"
SEG16GB_IVF_CSV  = f"{ROOT_RESULTS}/ablation_seg16384/MoRe_UPD_large_attidx_0/all_results_ivf.csv"

PLOTS_DIR = f"{ROOT_RESULTS}/ablation_seg16384/plots"

# Color palettes (4 shades each, one per selectivity level)
COLORS_DEFAULT = ["#FF6666", "#FF4444", "#FF2222", "#FF0000"]   # red  – 1 GB (default)
COLORS_SEG16   = ["#FFB866", "#FFA544", "#FF9222", "#FF8000"]   # orange – 16 GB

MARKERS    = ['.', '+', 'x', 's']
LINEWIDTHS = [0.6, 1.1, 1.6, 2.2]

K_VALUES = [1, 10, 40, 100]


def compute_averages_hnsw(df):
    """Average recall / runtime over individual queries AND all k values, keep ef_search axis."""
    df = df.copy()
    df = df[df['algorithm'] == 'milvus-hnsw']
    df['query_type'] = df['query_id'].apply(lambda x: 'movies' if x.startswith('qm') else 'reviews')
    averages = (
        df.groupby(['query_type', 'filter_id', 'filter_selectivity',
                     'ef_search', 'm'])[['recall', 'runtime']]
        .mean()
        .reset_index()
    )
    averages['throughput'] = 1.0 / averages['runtime']
    return averages


def compute_averages_ivf(df):
    """Average recall / runtime over individual queries AND all k values, keep probes axis."""
    df = df.copy()
    df = df[df['algorithm'] == 'milvus-ivfflat']
    df['query_type'] = df['query_id'].apply(lambda x: 'movies' if x.startswith('qm') else 'reviews')
    averages = (
        df.groupby(['query_type', 'filter_id', 'filter_selectivity',
                     'probes', 'clusters'])[['recall', 'runtime']]
        .mean()
        .reset_index()
    )
    averages['throughput'] = 1.0 / averages['runtime']
    return averages


def pick_selectivities(avg, query_type):
    """Return 4 selectivity levels using the same index logic as make_plots_results1.py."""
    sub = avg[avg['query_type'] == query_type]
    unique_sel = sorted(sub['filter_selectivity'].unique())
    indices = [i for i in [1, 3, 5, -1] if i < len(unique_sel)]
    return [unique_sel[i] for i in indices]


def plot_hnsw(ax, avg_default, avg_seg16, query_type):
    """QPS vs Recall for HNSW, comparing 1 GB vs 16 GB, averaged over all k."""
    M = 10  # fixed M parameter

    sub_def = avg_default[(avg_default['query_type'] == query_type) & (avg_default['m'] == M)]
    sub_16  = avg_seg16[(avg_seg16['query_type'] == query_type) & (avg_seg16['m'] == M)]

    sels = pick_selectivities(avg_default, query_type)
    handles, labels = [], []

    for idx, sel in enumerate(sels[:len(MARKERS)]):
        sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'

        # --- default 1 GB ---
        g = sub_def[sub_def['filter_selectivity'] == sel]
        if len(g):
            g = g.groupby('ef_search').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            g = g.sort_values('recall')
            lbl = f'1 GB  $\\sigma_g$≈{sel_approx}'
            h, = ax.plot(g['recall'], g['throughput'],
                         marker=MARKERS[idx], markersize=10, linewidth=LINEWIDTHS[idx],
                         color=COLORS_DEFAULT[idx], label=lbl)
            handles.append(h); labels.append(lbl)

        # --- 16 GB segment ---
        g = sub_16[sub_16['filter_selectivity'] == sel]
        if len(g):
            g = g.groupby('ef_search').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            g = g.sort_values('recall')
            lbl = f'16 GB $\\sigma_g$≈{sel_approx}'
            h, = ax.plot(g['recall'], g['throughput'],
                         marker=MARKERS[idx], markersize=10, linewidth=LINEWIDTHS[idx],
                         color=COLORS_SEG16[idx], label=lbl, linestyle='--')
            handles.append(h); labels.append(lbl)

    ax.set_xlim([0, 1])
    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--')
    query_cap = query_type.capitalize()
    ax.set_title(f'HNSW, {query_cap}, M={M}')

    return handles, labels


def plot_ivf(ax, avg_default, avg_seg16, query_type):
    """QPS vs Recall for IVF-Flat, comparing 1 GB vs 16 GB, averaged over all k."""
    CLUSTERS = 750  # fixed clusters parameter

    sub_def = avg_default[(avg_default['query_type'] == query_type) & (avg_default['clusters'] == CLUSTERS)]
    sub_16  = avg_seg16[(avg_seg16['query_type'] == query_type) & (avg_seg16['clusters'] == CLUSTERS)]

    sels = pick_selectivities(avg_default, query_type)
    handles, labels = [], []

    for idx, sel in enumerate(sels[:len(MARKERS)]):
        sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'

        # --- default 1 GB ---
        g = sub_def[sub_def['filter_selectivity'] == sel]
        if len(g):
            g = g.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            g = g.sort_values('recall')
            lbl = f'1 GB  $\\sigma_g$≈{sel_approx}'
            h, = ax.plot(g['recall'], g['throughput'],
                         marker=MARKERS[idx], markersize=10, linewidth=LINEWIDTHS[idx],
                         color=COLORS_DEFAULT[idx], label=lbl)
            handles.append(h); labels.append(lbl)

        # --- 16 GB segment ---
        g = sub_16[sub_16['filter_selectivity'] == sel]
        if len(g):
            g = g.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            g = g.sort_values('recall')
            lbl = f'16 GB $\\sigma_g$≈{sel_approx}'
            h, = ax.plot(g['recall'], g['throughput'],
                         marker=MARKERS[idx], markersize=10, linewidth=LINEWIDTHS[idx],
                         color=COLORS_SEG16[idx], label=lbl, linestyle='--')
            handles.append(h); labels.append(lbl)

    ax.set_xlim([0, 1])
    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--')
    query_cap = query_type.capitalize()
    ax.set_title(f'IVF-Flat, {query_cap}, nlist={CLUSTERS}')

    return handles, labels


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Load HNSW data ---
    df_hnsw_default = pd.read_csv(DEFAULT_HNSW_CSV)
    df_hnsw_seg16   = pd.read_csv(SEG16GB_HNSW_CSV)
    avg_hnsw_default = compute_averages_hnsw(df_hnsw_default)
    avg_hnsw_seg16   = compute_averages_hnsw(df_hnsw_seg16)

    # --- Load IVF data ---
    df_ivf_default = pd.read_csv(DEFAULT_IVF_CSV)
    df_ivf_seg16   = pd.read_csv(SEG16GB_IVF_CSV)
    avg_ivf_default = compute_averages_ivf(df_ivf_default)
    avg_ivf_seg16   = compute_averages_ivf(df_ivf_seg16)

    # --- Figure: 2×2 (HNSW top, IVF bottom; movies left, reviews right) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    all_handles, all_labels = [], []

    # Top row: HNSW
    for col_idx, qt in enumerate(['movies', 'reviews']):
        h, l = plot_hnsw(axes[0, col_idx], avg_hnsw_default, avg_hnsw_seg16, qt)
        if col_idx == 0:
            all_handles.extend(h)
            all_labels.extend(l)

    # Bottom row: IVF
    for col_idx, qt in enumerate(['movies', 'reviews']):
        h, l = plot_ivf(axes[1, col_idx], avg_ivf_default, avg_ivf_seg16, qt)

    # Unified legend below the plots
    fig.legend(all_handles, all_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.1),
               ncol=4, fontsize=22)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    for ext in ('pdf', 'png'):
        path = os.path.join(PLOTS_DIR, f'ablation_seg16gb_qps_vs_recall.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved in: {PLOTS_DIR}/ablation_seg16gb_qps_vs_recall.pdf/png")


if __name__ == '__main__':
    main()
