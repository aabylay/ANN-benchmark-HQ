"""
Workshop variant of VLDB paper plots: large datasets only.
Legend layout differs for Plot 1 (below, 3 cols: FAISS/Milvus/pgvector) and Plot 2b (below, 1 col).
- combined_throughput_vs_recall_by_selectivity (1 row x 4 cols) - from make_plots_results_ALL.py
- hnsw_throughput_vs_recall_by_m (1 row x 2 cols) - from make_plots_results1.py
- combined_hnsw_vs_ivf_comparison (2 rows x 3 cols) - from make_plots_hnsw_vs_ivf_comparison_ALL.py
- qps_recall_by_correlation (2 rows x 3 cols) - from create_results_with_correlation.py
- hnsw_throughput_vs_recall_by_selectivity_attidx (1 row x 4 cols) - from make_plots_attidx_comparison_ALL.py, pgvector only
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import os
import sys

# Allow importing create_results_with_correlation from same directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from create_results_with_correlation import (
        create_results_with_correlation,
        add_recall_from_all_results,
        plot_qps_recall_by_correlation,
        load_filter_stats,
    )
    CORRELATION_PLOT_AVAILABLE = True
except ImportError:
    CORRELATION_PLOT_AVAILABLE = False

# Large dataset only
DATASET_SIZE = 'large'

# Exact cardinalities for movies and reviews datasets
data_sizes = {
    "small": {"movies": "small", "reviews": "small"},
    "medium": {"movies": "medium", "reviews": "medium"},
    "large": {"movies": "large", "reviews": "large"},
}

# Algorithm names
algo_names_hnsw = {
    'milvus-hnsw': 'Milvus',
    'pgvector': 'pgvector',
    'hnsw(faiss)': 'FAISS',
}

algo_names_ivf = {
    'faiss-ivf': 'FAISS',
    'milvus-ivfflat': 'Milvus',
    'pgvector_ivf': 'pgvector',
}

hnsw_algo_names = {'milvus-hnsw': 'Milvus', 'pgvector': 'pgvector', 'hnsw(faiss)': 'FAISS'}
ivf_algo_names = {'milvus-ivfflat': 'Milvus', 'pgvector_ivf': 'pgvector', 'faiss-ivf': 'FAISS'}
system_order = ['FAISS', 'Milvus', 'pgvector']

mpl.rcParams['lines.linewidth'] = 2
plt.rc('font', family='serif', serif='DejaVu Serif', size=24)
plt.rc('mathtext', default='regular')

# Compact styling for 1-row x 4-col workshop figures (single-column paper width)
WORKSHOP_ROW4_FIGSIZE = (14, 2.6)
WORKSHOP_ROW4_STYLE = {
    'title': 16,
    'label': 14,
    'tick': 14,
    'legend': 14,
    'marker_scale': 0.6,
}

# Shared styling for the 2-row x 3-col ("3x2") workshop figures so they match.
# Same figsize => the same point size renders identically across both pictures.
WORKSHOP_3X2_FIGSIZE = (16, 8.5)
WORKSHOP_3X2_STYLE = {
    'title': 23,
    'label': 21,
    'tick': 19,
    'legend': 19,
}
# Horizontal gap between columns, as a fraction of average axis width (subplots_adjust wspace).
WORKSHOP_3X2_WSPACE = 0.20


def finalize_workshop_row4_figure(fig, axes, legend_handles, legend_labels, legend_ncol, ylabel='QPS', legend_footnote=None):
    """Shrink text and enlarge plot area for 1x4 workshop figures."""
    style = WORKSHOP_ROW4_STYLE
    axes_flat = np.atleast_1d(axes).flat
    for i, ax in enumerate(axes_flat):
        ax.set_title(ax.get_title(), fontsize=style['title'], pad=2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=style['label'], labelpad=1)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=style['label'], labelpad=1)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=style['tick'], pad=1, length=3)
        for line in ax.get_lines():
            ms = line.get_markersize()
            if ms and ms > 0:
                line.set_markersize(max(3, ms * style['marker_scale']))
    leg = None
    if legend_handles:
        leg = fig.legend(
            legend_handles, legend_labels,
            loc='upper center', bbox_to_anchor=(0.5, 0.03 if legend_footnote else 0.01),
            ncol=legend_ncol, fontsize=style['legend'],
            frameon=True, handlelength=2.0, handletextpad=0.4,
            columnspacing=1.0, borderpad=0.2,
        )
    fig.subplots_adjust(left=0.05, right=0.995, top=0.86, bottom=0.28 if legend_footnote else 0.24, wspace=0.20)
    
    """
    if legend_footnote:
        if leg is not None:
            # Place the footnote directly below the legend box (not above it).
            fig.canvas.draw()
            _lb = leg.get_window_extent().transformed(fig.transFigure.inverted())
            fig.text(0.5, _lb.y0 - 0.02, legend_footnote, ha='center', va='top', fontsize=style['legend'])
        else:
            fig.text(0.5, 0.005, legend_footnote, ha='center', va='bottom', fontsize=style['legend'])
    """

def apply_workshop_3x2_style(axes, ylabel='QPS'):
    """Force identical title/label/tick font sizes across the 2x3 workshop figures.

    The y-axis label (QPS) is shown only on the first column of each row, like the
    1x4 figures show it only on the first axes; the y-tick numbers stay on every
    subplot. Call this after the per-axes plotting helpers (which set titles/labels
    at the global rcParams size) but before tight_layout, so the layout accounts for
    the final font sizes. Legend size is set via WORKSHOP_3X2_STYLE['legend'].
    """
    style = WORKSHOP_3X2_STYLE
    for row in np.atleast_2d(axes):
        for col_idx, ax in enumerate(np.atleast_1d(row)):
            ax.set_title(ax.get_title(), fontsize=style['title'])
            ax.set_xlabel(ax.get_xlabel(), fontsize=style['label'])
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=style['label'])
            else:
                ax.set_ylabel('')
            ax.tick_params(axis='both', labelsize=style['tick'])


def build_attidx_legend_entries(legend_metadata, markers_sel_att, attidx_colors, attidx_linestyles, attidx_linewidth_mult):
    """Build att_idx legend ordered for ncol=2 column-major layout.

    Left column: att_idx=0 at each selectivity (top to bottom).
    Right column: att_idx=1 at each selectivity (top to bottom).
    """
    legend_data = {}
    for _handle, att_idx, sel_idx, sel_value, qt in legend_metadata:
        key = (att_idx, sel_idx)
        if key not in legend_data:
            legend_data[key] = {'sel_values': {}}
        legend_data[key]['sel_values'][qt] = sel_value

    sel_indices = sorted({sel_idx for _, sel_idx in legend_data.keys()})
    n_sel = len(sel_indices)
    legend_handles, legend_labels = [], []

    for att_idx in [0, 1]:
        for sel_idx in sel_indices:
            if (att_idx, sel_idx) not in legend_data:
                continue
            sel_dict = legend_data[(att_idx, sel_idx)]['sel_values']
            sel_strs = []
            for qt, short in [('movies', 'M'), ('reviews', 'R')]:
                if qt in sel_dict:
                    sel_val = sel_dict[qt]
                    sel_str = f'{sel_val:.2f}' if sel_val >= 0.01 else f'{sel_val:.3f}'
                    sel_strs.append(f'{short}: {sel_str}')
            sel_label = '{' + ', '.join(sel_strs) + '}'
            marker = markers_sel_att[sel_idx] if sel_idx < len(markers_sel_att) else 'o'
            lw_mult = attidx_linewidth_mult[sel_idx] if sel_idx < len(attidx_linewidth_mult) else 2.0
            if n_sel == 1:
                lw_mult = 2.0
            linewidth = 2 * lw_mult
            legend_handles.append(
                Line2D([0], [0], color=attidx_colors[att_idx], marker=marker,
                       linestyle=attidx_linestyles[att_idx], linewidth=linewidth * 0.5,
                       markersize=(6 if att_idx == 0 else 8))
            )
            if att_idx: att_text = 'with attr. idx.'
            else: att_text = 'w/o attr. idx.'
            legend_labels.append(f'{att_text}, $\\sigma_g$={sel_label}')
    return legend_handles, legend_labels


def compute_averages_hnsw(df):
    df['query_type'] = df['query_id'].apply(lambda x: 'movies' if x.startswith('qm') else 'reviews')
    averages = df.groupby(['query_type', 'filter_id', 'filter_selectivity', 'k', 'ef_search', 'algorithm', 'm'])[['recall', 'runtime']].mean().reset_index()
    averages['throughput'] = 1 / averages['runtime']
    return averages


def compute_averages_ivf(df):
    df['query_type'] = df['query_id'].apply(lambda x: 'movies' if x.startswith('qm') else 'reviews')
    averages = df.groupby(['query_type', 'filter_id', 'filter_selectivity', 'k', 'probes', 'algorithm', 'clusters'])[['recall', 'runtime']].mean().reset_index()
    averages['throughput'] = 1 / averages['runtime']
    return averages


# =============================================================================
# Plot 1: combined_throughput_vs_recall_by_selectivity (from make_plots_results_ALL.py)
# 1 row x 4 cols: movies-HNSW, movies-IVF, reviews-HNSW, reviews-IVF
# =============================================================================

def plot_throughput_vs_recall_by_selectivity_hnsw(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Throughput vs recall by selectivity for HNSW (k=10, m=10)"""
    markers_sel = ['.', '+', 'x', 's']
    linewidths = [0.6, 1.1, 1.6, 2.2]
    colors_sel = [
        ["#FF6666", "#FF4444", "#FF2222", "#FF0000"],
        ["#6666FF", "#4444FF", "#2222FF", "#0000FF"],
        ["#66FF66", "#44FF44", "#22FF22", "#00FF00"],
    ]
    sub_qt = averages[(averages['query_type'] == query_type) & (averages['k'] == 10) & (averages['m'] == 10)]
    unique_sel = sorted(sub_qt['filter_selectivity'].unique())
    unique_sel = [unique_sel[i] for i in [1, 3, 5, -1] if i < len(unique_sel)]
    handles, labels, metadata = [], [], []
    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel = sub_qt[sub_qt['filter_selectivity'] == sel]
        for algo in sub_sel['algorithm'].unique():
            algo_idx = {'milvus-hnsw': 0, 'pgvector': 1, 'hnsw(faiss)': 2}.get(algo, 2)
            sub = sub_sel[sub_sel['algorithm'] == algo]
            grouped = sub.groupby('ef_search').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label = f'{algo_names_hnsw[algo]} $\\sigma_g$≈{sel_approx}'
            line, = ax.plot(grouped['recall'], grouped['throughput'],
                            marker=markers_sel[idx], markersize=10, linewidth=linewidths[idx],
                            color=colors_sel[algo_idx][idx], label=label)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
                metadata.append((line, algo, idx, sel, dataset_size))
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1.5, 10**3.5])
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}, HNSW')
    ax.set_yscale('log')
    if len(sub_qt) > 0:
        max_throughput = max(sub_qt['throughput'])
        ax.set_yticks([10**i for i in range(1, int(np.ceil(np.log10(max_throughput))))])
    ax.grid(True, which="both", ls="--")
    return handles, labels, metadata


def plot_throughput_vs_recall_by_selectivity_ivf(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Throughput vs recall by selectivity for IVF (k=10) - darker colors like HNSW vs IVF comparison"""
    markers_sel = ['.', '+', 'x', 's']
    linewidths = [0.6, 1.1, 1.6, 2.2]
    # Darker/greyer versions for IVF (matching system_colors_ivf in hnsw_vs_ivf_comparison)
    colors_sel = [
        ["#FF6666", "#FF4444", "#FF2222", "#FF0000"],
        ["#6666FF", "#4444FF", "#2222FF", "#0000FF"],
        ["#66FF66", "#44FF44", "#22FF22", "#00FF00"],
    ]
    sub_qt = averages[(averages['query_type'] == query_type) & (averages['k'] == 10)]
    unique_sel = sorted(sub_qt['filter_selectivity'].unique())
    unique_sel = [unique_sel[i] for i in [1, 3, 5, -1] if i < len(unique_sel)]
    handles, labels, metadata = [], [], []
    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel = sub_qt[sub_qt['filter_selectivity'] == sel]
        for algo in sub_sel['algorithm'].unique():
            algo_idx = {'milvus-ivfflat': 0, 'pgvector_ivf': 1, 'faiss-ivf': 2}.get(algo, 2)
            sub = sub_sel[sub_sel['algorithm'] == algo]
            grouped = sub.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label = f'{algo_names_ivf[algo]} $\\sigma_g$≈{sel_approx}'
            line, = ax.plot(grouped['recall'], grouped['throughput'],
                            marker=markers_sel[idx], markersize=10, linewidth=linewidths[idx],
                            color=colors_sel[algo_idx][idx], label=label)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
                metadata.append((line, algo, idx, sel, dataset_size))
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}, IVFFlat')
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**0.2, 10**3.5])
    ax.set_yticks([10**i for i in range(1, 4)])
    ax.grid(True, which="both", ls="--")
    return handles, labels, metadata


def create_unified_legend_throughput_recall(all_metadata_hnsw, all_metadata_ivf, algo_names):
    """Create unified legend for throughput vs recall by selectivity"""
    legend_data = {}
    for handle, algo, sel_idx, sel_value, dataset_size in all_metadata_hnsw:
        key = (algo, sel_idx, 'hnsw')
        if key not in legend_data:
            legend_data[key] = {'handle': handle, 'sel_values': {}, 'algo_name': algo_names.get(algo, algo)}
        legend_data[key]['sel_values'][dataset_size] = sel_value
    for handle, algo, sel_idx, sel_value, dataset_size in all_metadata_ivf:
        key = (algo, sel_idx, 'ivf')
        if key not in legend_data:
            legend_data[key] = {'handle': handle, 'sel_values': {}, 'algo_name': algo_names.get(algo, algo)}
        legend_data[key]['sel_values'][dataset_size] = sel_value
    legend_handles, legend_labels = [], []
    sorted_keys = sorted(legend_data.keys(), key=lambda x: (legend_data[x]['algo_name'], x[1]))
    for key in sorted_keys:
        data = legend_data[key]
        sel_dict = data['sel_values']
        sel_strs = []
        for ds in ['small', 'medium', 'large']:
            if ds in sel_dict:
                sel_val = sel_dict[ds]
                sel_str = f'{sel_val:.2f}' if sel_val >= 0.01 else f'{sel_val:.3f}'
                sel_strs.append(f'{ds[0].upper()}: {sel_str}')
        sel_label = '{' + ', '.join(sel_strs) + '}'
        legend_handles.append(data['handle'])
        legend_labels.append(f'{data["algo_name"]} $\\sigma_g$={sel_label}')
    return legend_handles, legend_labels


def create_unified_legend_throughput_recall_by_system_columns(all_metadata_hnsw, all_metadata_ivf, algo_names):
    """Legend entries ordered for ncol=4 with matplotlib column-first fill.

    Columns: one selectivity level per column.
    Rows: systems (FAISS, Milvus, pgvector) within each column.
    """
    legend_data = {}
    for handle, algo, sel_idx, sel_value, dataset_size in all_metadata_hnsw:
        key = (sel_idx, algo)
        if key not in legend_data:
            legend_data[key] = {'handle': handle, 'sel_values': {}, 'algo_name': algo_names.get(algo, algo)}
        legend_data[key]['sel_values'][dataset_size] = sel_value
    for handle, algo, sel_idx, sel_value, dataset_size in all_metadata_ivf:
        key = (sel_idx, algo)
        if key not in legend_data:
            legend_data[key] = {'handle': handle, 'sel_values': {}, 'algo_name': algo_names.get(algo, algo)}
        legend_data[key]['sel_values'][dataset_size] = sel_value

    algo_by_system = {name: algo for algo, name in algo_names.items()}
    sel_indices = sorted({sel_idx for sel_idx, _ in legend_data.keys()})
    legend_handles, legend_labels = [], []
    # Selectivity-first order: matplotlib fills column-by-column, so each column is
    # one selectivity level and the rows within it are the systems.
    for sel_idx in sel_indices:
        for sys_name in system_order:
            algo = algo_by_system.get(sys_name)
            if algo is None or (sel_idx, algo) not in legend_data:
                continue
            data = legend_data[(sel_idx, algo)]
            sel_dict = data['sel_values']
            sel_strs = []
            for ds in ['small', 'medium', 'large']:
                if ds in sel_dict:
                    sel_val = sel_dict[ds]
                    sel_str = f'{sel_val:.2f}' if sel_val >= 0.01 else f'{sel_val:.3f}'
                    sel_strs.append(sel_str)
            sel_label = sel_strs[0] if len(sel_strs) == 1 else '{' + ', '.join(sel_strs) + '}'
            legend_handles.append(data['handle'])
            legend_labels.append(f'{sys_name}, $\\sigma_g$={sel_label}')
    return legend_handles, legend_labels


# =============================================================================
# Plot 2: hnsw_throughput_vs_recall_by_m (from make_plots_results1.py)
# 1 row x 2 cols: [movies], [reviews]
# =============================================================================

def plot_throughput_vs_recall_by_ef_search(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Throughput vs recall by ef_search, different m values"""
    markers = ['D', 'o', 's']
    colors_sel = [
        ["#FF8888", "#FF5555", "#FF3333", "#FF0000"],
        ["#8888FF", "#5555FF", "#3333FF", "#0000FF"],
        ["#66FF66", "#44FF44", "#22FF22", "#00FF00"],
    ]
    sub_qt = averages[(averages['query_type'] == query_type) & (averages['k'] == 10)]
    m_values = [5, 10, 15]
    handles, labels = [], []
    for algo in sub_qt['algorithm'].unique():
        algo_idx = {'milvus-hnsw': 0, 'pgvector': 1, 'hnsw(faiss)': 2}.get(algo, 2)
        for i, m in enumerate(m_values):
            sub = sub_qt[(sub_qt['algorithm'] == algo) & (sub_qt['m'] == m)]
            if len(sub) == 0:
                continue
            grouped = sub.groupby('ef_search').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='recall')
            label = f'{algo_names_hnsw[algo]}, M={m}'
            line, = ax.plot(grouped['recall'], grouped['throughput'], marker=markers[i], markersize=8,
                            color=colors_sel[algo_idx][i], label=label, linewidth=m * 0.2)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}')
    ax.set_yscale('log')
    if len(sub_qt) > 0:
        max_throughput = max(sub_qt['throughput'])
        yticks = [10**i for i in range(1, int(np.ceil(np.log10(max_throughput))) + 1)] + [10**3.5]
        ax.set_yticks(yticks)
    ax.grid(True, which="both", ls="--")
    return handles, labels


# =============================================================================
# Plot 3: combined_hnsw_vs_ivf_comparison (from make_plots_hnsw_vs_ivf_comparison_ALL.py)
# 2 rows x 3 cols: rows=[movies, reviews], cols=[FAISS, Milvus, pgvector]
# =============================================================================

def plot_hnsw_ivf_comparison(ax, hnsw_averages, ivf_averages, dataset_size, query_type='movies', system_name='Milvus', return_handles_labels=False):
    """HNSW vs IVF QPS vs Recall by selectivity for a specific system"""
    markers_sel = ['.', '+', 'x', 's']
    system_colors = {'pgvector': '#0000FF', 'Milvus': '#FF0000', 'FAISS': '#00FF00'}
    system_colors_ivf = {'pgvector': '#6666AA', 'Milvus': '#AA6666', 'FAISS': '#66AA66'}
    system_color = system_colors.get(system_name, '#000000')
    system_color_ivf = system_colors_ivf.get(system_name, '#666666')
    base_linewidth = 2
    linewidth_multipliers = [0.8, 0.9, 1.4, 2]
    hnsw_algo_map = {'Milvus': 'milvus-hnsw', 'pgvector': 'pgvector', 'FAISS': 'hnsw(faiss)'}
    ivf_algo_map = {'Milvus': 'milvus-ivfflat', 'pgvector': 'pgvector_ivf', 'FAISS': 'faiss-ivf'}
    hnsw_algo = hnsw_algo_map.get(system_name)
    ivf_algo = ivf_algo_map.get(system_name)

    hnsw_sub = hnsw_averages[(hnsw_averages['query_type'] == query_type) &
                             (hnsw_averages['k'] == 10) &
                             (hnsw_averages['m'] == 10) &
                             (hnsw_averages['algorithm'] == hnsw_algo)]
    ivf_sub = ivf_averages[(ivf_averages['query_type'] == query_type) &
                           (ivf_averages['k'] == 10) &
                           (ivf_averages['algorithm'] == ivf_algo)]

    hnsw_sel = sorted(hnsw_sub['filter_selectivity'].unique()) if len(hnsw_sub) > 0 else []
    ivf_sel = sorted(ivf_sub['filter_selectivity'].unique()) if len(ivf_sub) > 0 else []
    all_sel = sorted(set(hnsw_sel + ivf_sel))
    unique_sel = [all_sel[i] for i in [1, 3, 5, -1] if i < len(all_sel)]

    handles, labels, metadata = [], [], []

    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        linestyle = '-' if idx != 0 else '--'
        sub_sel = hnsw_sub[hnsw_sub['filter_selectivity'] == sel]
        if len(sub_sel) > 0:
            grouped = sub_sel.groupby('ef_search').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label = f'HNSW $\\sigma_g$≈{sel_approx}'
            linewidth = base_linewidth * linewidth_multipliers[idx]
            line, = ax.plot(grouped['recall'], grouped['throughput'], marker=markers_sel[idx],
                            color=system_color, label=label, linestyle=linestyle, linewidth=linewidth)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
                metadata.append((line, 'HNSW', idx, sel, query_type))

    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        linestyle = '-' if idx != 0 else '--'
        sub_sel = ivf_sub[ivf_sub['filter_selectivity'] == sel]
        if len(sub_sel) > 0:
            grouped = sub_sel.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label = f'IVF $\\sigma_g$≈{sel_approx}'
            linewidth = base_linewidth * linewidth_multipliers[idx]
            line, = ax.plot(grouped['recall'], grouped['throughput'], marker=markers_sel[idx],
                            color=system_color_ivf, label=label, linestyle=linestyle, linewidth=linewidth)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
                metadata.append((line, 'IVF', idx, sel, query_type))

    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    if system_name == 'Milvus':
        ax.set_ylim([10**1, 10**2.5])
        ax.set_xlim([0.6, 1])
    ylim = ax.get_ylim()
    ax.axvline(x=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
    ax.set_ylim(ylim)
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    ax.set_title(f'{query_type.capitalize()}, {system_name}')
    ax.set_yscale('log')
    all_throughput = []
    if len(hnsw_sub) > 0:
        all_throughput.extend(hnsw_sub['throughput'].values)
    if len(ivf_sub) > 0:
        all_throughput.extend(ivf_sub['throughput'].values)
    if len(all_throughput) > 0:
        max_throughput = max(all_throughput)
        ax.set_yticks([10**i for i in range(1, int(np.ceil(np.log10(max_throughput))))])
        if system_name == 'pgvector':
            ax.set_yticks([10**1, 10**2, 10**3])
    ax.grid(True, which="both", ls="--")
    return handles, labels, metadata


# =============================================================================
# Plot: throughput vs recall by selectivity att_idx (from make_plots_attidx_comparison_ALL.py)
# pgvector only, large dataset, 1 row x 4 cols: movies-HNSW, movies-IVF, reviews-HNSW, reviews-IVF
# =============================================================================

def plot_throughput_vs_recall_attidx_hnsw(ax, averages_0, averages_1, dataset_size, query_type='movies', algorithm='pgvector', return_handles_labels=False):
    """Throughput vs recall by selectivity comparing att_idx=0 vs att_idx=1 for HNSW (k=10, m=10)"""
    markers_sel = ['D', 'X', 'o']
    algo_display = 'Milvus' if algorithm == 'milvus-hnsw' else 'pgvector'
    if algo_display == 'pgvector':
        system_color = "#0000FF"
        system_color2 = "#8888FF"
    else:
        system_color = "#FF0000"
        system_color2 = "#FF8888"
    base_linewidth = 2

    if averages_0 is None:
        averages_0 = pd.DataFrame()
    if averages_1 is None:
        averages_1 = pd.DataFrame()

    if len(averages_0) > 0:
        sub_qt_0 = averages_0[(averages_0['query_type'] == query_type) & (averages_0['k'] == 10) &
                             (averages_0['m'] == 10) & (averages_0['algorithm'] == algorithm)]
    else:
        sub_qt_0 = pd.DataFrame()

    if len(averages_1) > 0:
        sub_qt_1 = averages_1[(averages_1['query_type'] == query_type) & (averages_1['k'] == 10) &
                             (averages_1['m'] == 10) & (averages_1['algorithm'] == algorithm)]
    else:
        sub_qt_1 = pd.DataFrame()

    unique_sel_0 = sorted(sub_qt_0['filter_selectivity'].unique()) if len(sub_qt_0) > 0 else []
    unique_sel_1 = sorted(sub_qt_1['filter_selectivity'].unique()) if len(sub_qt_1) > 0 else []
    unique_sel = sorted(set(unique_sel_0) & set(unique_sel_1)) if (unique_sel_0 and unique_sel_1) else sorted(set(unique_sel_0) | set(unique_sel_1))
    unique_sel = [unique_sel[i] for i in [1, 4, -1] if i < len(unique_sel)]

    handles, labels, metadata = [], [], []
    n_sel = len(unique_sel[:len(markers_sel)])
    linewidth_multipliers = []
    for i in range(n_sel):
        mult = 2.0 if (n_sel == 1 or i == n_sel - 1) else (0.6 if i == 0 else 1.3)
        linewidth_multipliers.append(mult)

    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel_0 = sub_qt_0[sub_qt_0['filter_selectivity'] == sel] if len(sub_qt_0) > 0 else pd.DataFrame()
        sub_sel_1 = sub_qt_1[sub_qt_1['filter_selectivity'] == sel] if len(sub_qt_1) > 0 else pd.DataFrame()
        linewidth = base_linewidth * linewidth_multipliers[idx]

        if len(sub_sel_0) > 0:
            grouped_0 = sub_sel_0.groupby('ef_search').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped_0 = grouped_0.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label_0 = f'att_idx=0 $\\sigma_g$≈{sel_approx}'
            line_0, = ax.plot(grouped_0['recall'], grouped_0['throughput'],
                             marker=markers_sel[idx], markersize=10, color=system_color,
                             linestyle='-', label=label_0, linewidth=linewidth)
            if return_handles_labels:
                handles.append(line_0)
                labels.append(label_0)
                metadata.append((line_0, 0, idx, sel, query_type))

        if len(sub_sel_1) > 0:
            grouped_1 = sub_sel_1.groupby('ef_search').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped_1 = grouped_1.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label_1 = f'att_idx=1 $\\sigma_g$≈{sel_approx}'
            line_1, = ax.plot(grouped_1['recall'], grouped_1['throughput'],
                             marker=markers_sel[idx], markersize=14, color=system_color2,
                             linestyle='--', label=label_1, linewidth=linewidth)
            if return_handles_labels:
                handles.append(line_1)
                labels.append(label_1)
                metadata.append((line_1, 1, idx, sel, query_type))

    ax.axvline(x=1.0, color='#000000', linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}')
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    if algo_display == 'pgvector':
        ax.set_xlim([0, 1.05])
        if len(sub_qt_0) > 0 or len(sub_qt_1) > 0:
            max_throughput = max(
                max(sub_qt_0['throughput']) if len(sub_qt_0) > 0 else 0,
                max(sub_qt_1['throughput']) if len(sub_qt_1) > 0 else 0
            )
            yticks = [10**i for i in range(1, int(np.ceil(np.log10(max_throughput))))]
            yticks.append(10**3.5)
            ax.set_yticks(yticks)
    ax.grid(True, which="both", ls="--", zorder=0)
    return handles, labels, metadata


def plot_throughput_vs_recall_attidx_ivf(ax, averages_0, averages_1, dataset_size, query_type='movies', algorithm='pgvector_ivf', return_handles_labels=False):
    """Throughput vs recall by selectivity comparing att_idx=0 vs att_idx=1 for IVF (k=10)"""
    markers_sel = ['D', 'X', 'o']
    algo_display = 'Milvus' if algorithm == 'milvus-ivfflat' else 'pgvector'
    if algo_display == 'pgvector':
        system_color = "#0000FF"
        system_color2 = "#8888FF"
    else:
        system_color = "#FF0000"
        system_color2 = "#FF8888"
    base_linewidth = 2

    if averages_0 is None:
        averages_0 = pd.DataFrame()
    if averages_1 is None:
        averages_1 = pd.DataFrame()

    if len(averages_0) > 0:
        sub_qt_0 = averages_0[(averages_0['query_type'] == query_type) & (averages_0['k'] == 10) &
                             (averages_0['algorithm'] == algorithm)]
    else:
        sub_qt_0 = pd.DataFrame()

    if len(averages_1) > 0:
        sub_qt_1 = averages_1[(averages_1['query_type'] == query_type) & (averages_1['k'] == 10) &
                             (averages_1['algorithm'] == algorithm)]
    else:
        sub_qt_1 = pd.DataFrame()

    unique_sel_0 = sorted(sub_qt_0['filter_selectivity'].unique()) if len(sub_qt_0) > 0 else []
    unique_sel_1 = sorted(sub_qt_1['filter_selectivity'].unique()) if len(sub_qt_1) > 0 else []
    unique_sel = sorted(set(unique_sel_0) & set(unique_sel_1)) if (unique_sel_0 and unique_sel_1) else sorted(set(unique_sel_0) | set(unique_sel_1))
    unique_sel = [unique_sel[i] for i in [1, 4, -1] if i < len(unique_sel)]

    handles, labels, metadata = [], [], []
    n_sel = len(unique_sel[:len(markers_sel)])
    linewidth_multipliers = []
    for i in range(n_sel):
        mult = 2.0 if (n_sel == 1 or i == n_sel - 1) else (0.6 if i == 0 else 1.3)
        linewidth_multipliers.append(mult)

    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel_0 = sub_qt_0[sub_qt_0['filter_selectivity'] == sel] if len(sub_qt_0) > 0 else pd.DataFrame()
        sub_sel_1 = sub_qt_1[sub_qt_1['filter_selectivity'] == sel] if len(sub_qt_1) > 0 else pd.DataFrame()
        linewidth = base_linewidth * linewidth_multipliers[idx]

        if len(sub_sel_0) > 0:
            grouped_0 = sub_sel_0.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped_0 = grouped_0.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label_0 = f'att_idx=0 $\\sigma_g$≈{sel_approx}'
            line_0, = ax.plot(grouped_0['recall'], grouped_0['throughput'],
                             marker=markers_sel[idx], markersize=10, color=system_color,
                             linestyle='-', label=label_0, linewidth=linewidth)
            if return_handles_labels:
                handles.append(line_0)
                labels.append(label_0)
                metadata.append((line_0, 0, idx, sel, query_type))

        if len(sub_sel_1) > 0:
            grouped_1 = sub_sel_1.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped_1 = grouped_1.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label_1 = f'att_idx=1 $\\sigma_g$≈{sel_approx}'
            line_1, = ax.plot(grouped_1['recall'], grouped_1['throughput'],
                             marker=markers_sel[idx], markersize=14, color=system_color2,
                             linestyle='--', label=label_1, linewidth=linewidth)
            if return_handles_labels:
                handles.append(line_1)
                labels.append(label_1)
                metadata.append((line_1, 1, idx, sel, query_type))

    ax.axvline(x=1.0, color='#000000', linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}')
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    if algo_display == 'pgvector':
        ax.set_xlim([0, 1.05])
        ax.set_ylim([10**0, 10**3.5])
        ax.set_yticks([10**i for i in range(0, 4)])
    ax.grid(True, which="both", ls="--", zorder=0)
    return handles, labels, metadata


def main():
    root_results = '/home/abylay/ann-benchmarks-HQ/results_prev'
    plots_dir = f"{root_results}/MoRe_UPD_plots_workshop"
    os.makedirs(plots_dir, exist_ok=True)

    # Load large dataset only
    hnsw_csv = f"{root_results}/MoRe_UPD_{DATASET_SIZE}_attidx_0/all_results_hnsw.csv"
    ivf_csv = f"{root_results}/MoRe_UPD_{DATASET_SIZE}_attidx_0/all_results_ivf.csv"

    try:
        df_hnsw = pd.read_csv(hnsw_csv)
        averages_hnsw = compute_averages_hnsw(df_hnsw)
    except FileNotFoundError:
        print(f"Error: HNSW CSV not found at {hnsw_csv}")
        averages_hnsw = None

    try:
        df_ivf = pd.read_csv(ivf_csv)
        averages_ivf = compute_averages_ivf(df_ivf)
    except FileNotFoundError:
        print(f"Error: IVF CSV not found at {ivf_csv}")
        averages_ivf = None

    # -------------------------------------------------------------------------
    # Plot 1: combined_throughput_vs_recall_by_selectivity - 1 row x 4 cols (large only)
    # Cols: movies-HNSW, movies-IVF, reviews-HNSW, reviews-IVF | Legend below, 4 cols: one selectivity level per column
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 4, figsize=WORKSHOP_ROW4_FIGSIZE)
    all_metadata_hnsw = []

    for col_idx, (query_type, index_type) in enumerate([
        ('movies', 'hnsw'), ('movies', 'ivf'), ('reviews', 'hnsw'), ('reviews', 'ivf'),
    ]):
        if index_type == 'hnsw' and averages_hnsw is not None:
            handles, labels, metadata = plot_throughput_vs_recall_by_selectivity_hnsw(
                axes1[col_idx], averages_hnsw, DATASET_SIZE, query_type, return_handles_labels=True)
            if query_type == 'movies':
                all_metadata_hnsw.extend(metadata)
        elif index_type == 'ivf' and averages_ivf is not None:
            plot_throughput_vs_recall_by_selectivity_ivf(
                axes1[col_idx], averages_ivf, DATASET_SIZE, query_type, return_handles_labels=False)

    # Unified legend below: 4 columns, one selectivity level each (use HNSW metadata from movies row)
    legend_handles, legend_labels = create_unified_legend_throughput_recall_by_system_columns(
        all_metadata_hnsw, [], algo_names_hnsw)
    finalize_workshop_row4_figure(fig1, axes1, legend_handles, legend_labels, legend_ncol=4)
    _save = dict(dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(plots_dir, 'workshop_combined_throughput_vs_recall_by_selectivity.pdf'), **_save)
    plt.savefig(os.path.join(plots_dir, 'workshop_combined_throughput_vs_recall_by_selectivity.png'), **_save)
    plt.close()
    print(f"Plot 1 saved: {plots_dir}/workshop_combined_throughput_vs_recall_by_selectivity.pdf/png")

    # -------------------------------------------------------------------------
    # Plot 2: hnsw_throughput_vs_recall_by_m - 1 row x 2 cols (large only)
    # Cols: movies, reviews
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 4))  # 1 row, 2 cols
    all_handles, all_labels = [], []
    if averages_hnsw is not None:
        for col_idx, query_type in enumerate(['movies', 'reviews']):
            handles, labels = plot_throughput_vs_recall_by_ef_search(
                axes2[col_idx], averages_hnsw, DATASET_SIZE, query_type, return_handles_labels=True)
            if col_idx == 0:
                all_handles, all_labels = handles, labels

    algo_groups = {}
    for handle, label in zip(all_handles, all_labels):
        algo_name = label.split(',')[0]
        if algo_name not in algo_groups:
            algo_groups[algo_name] = []
        algo_groups[algo_name].append((handle, label))
    sorted_handles = []
    sorted_labels = []
    for algo in sorted(algo_groups.keys()):
        for handle, label in algo_groups[algo]:
            sorted_handles.append(handle)
            sorted_labels.append(label)
    fig2.legend(sorted_handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.45, 0.02), ncol=3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(plots_dir, 'vldb_hnsw_throughput_vs_recall_by_m.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'vldb_hnsw_throughput_vs_recall_by_m.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 2 saved: {plots_dir}/vldb_hnsw_throughput_vs_recall_by_m.pdf/png")

    # -------------------------------------------------------------------------
    # Plot 2b: throughput vs recall by selectivity att_idx (pgvector, large only)
    # 1 row x 4 cols: movies-HNSW, movies-IVF, reviews-HNSW, reviews-IVF | Legend below
    # -------------------------------------------------------------------------
    hnsw_attidx_0 = hnsw_attidx_1 = ivf_attidx_0 = ivf_attidx_1 = None
    try:
        df = pd.read_csv(f"{root_results}/MoRe_UPD_{DATASET_SIZE}_attidx_0/all_results_hnsw.csv")
        hnsw_attidx_0 = compute_averages_hnsw(df)
    except FileNotFoundError:
        pass
    try:
        df = pd.read_csv(f"{root_results}/MoRe_UPD_{DATASET_SIZE}_attidx_1/all_results_hnsw.csv")
        hnsw_attidx_1 = compute_averages_hnsw(df)
    except FileNotFoundError:
        pass
    try:
        df = pd.read_csv(f"{root_results}/MoRe_UPD_{DATASET_SIZE}_attidx_0/all_results_ivf.csv")
        ivf_attidx_0 = compute_averages_ivf(df)
    except FileNotFoundError:
        pass
    try:
        df = pd.read_csv(f"{root_results}/MoRe_UPD_{DATASET_SIZE}_attidx_1/all_results_ivf.csv")
        ivf_attidx_1 = compute_averages_ivf(df)
    except FileNotFoundError:
        pass

    has_hnsw = hnsw_attidx_0 is not None or hnsw_attidx_1 is not None
    has_ivf = ivf_attidx_0 is not None or ivf_attidx_1 is not None
    if has_hnsw or has_ivf:
        fig_attidx, axes_attidx = plt.subplots(1, 4, figsize=WORKSHOP_ROW4_FIGSIZE)
        legend_metadata = []
        attidx_colors = {0: '#0000FF', 1: '#8888FF'}
        attidx_linestyles = {0: '-', 1: '--'}
        markers_sel_att = ['D', 'X', 'o']
        attidx_linewidth_mult = [0.6, 1.3, 2.0]

        for col_idx, (query_type, index_type) in enumerate([
            ('movies', 'hnsw'), ('movies', 'ivf'), ('reviews', 'hnsw'), ('reviews', 'ivf'),
        ]):
            ax = axes_attidx[col_idx]
            if index_type == 'hnsw' and has_hnsw:
                handles, labels, metadata = plot_throughput_vs_recall_attidx_hnsw(
                    ax, hnsw_attidx_0, hnsw_attidx_1,
                    DATASET_SIZE, query_type, algorithm='pgvector', return_handles_labels=True
                )
                legend_metadata.extend(metadata)
                ax.set_title(f'{query_type.capitalize()}, HNSW')
            elif index_type == 'ivf' and has_ivf:
                plot_throughput_vs_recall_attidx_ivf(
                    ax, ivf_attidx_0, ivf_attidx_1,
                    DATASET_SIZE, query_type, algorithm='pgvector_ivf', return_handles_labels=False
                )
                ax.set_title(f'{query_type.capitalize()}, IVFFlat')

        # Unified legend below: 2 columns (att_idx=0 | att_idx=1), selectivity top-to-bottom
        legend_handles, legend_labels = build_attidx_legend_entries(
            legend_metadata, markers_sel_att, attidx_colors, attidx_linestyles, attidx_linewidth_mult
        ) if legend_metadata else ([], [])

        finalize_workshop_row4_figure(
            fig_attidx, axes_attidx, legend_handles, legend_labels, legend_ncol=2,
            legend_footnote='M — Movies, R — Reviews')
        _save_att = dict(dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(plots_dir, 'workshop_hnsw_throughput_vs_recall_by_selectivity_attidx_pgvector.pdf'), **_save_att)
        plt.savefig(os.path.join(plots_dir, 'workshop_hnsw_throughput_vs_recall_by_selectivity_attidx_pgvector.png'), **_save_att)
        plt.close()
        print(f"Plot 2b saved: {plots_dir}/workshop_hnsw_throughput_vs_recall_by_selectivity_attidx_pgvector.pdf/png")
    else:
        print("Plot 2b (attidx pgvector) skipped: missing attidx 0 or 1 data.")

    # -------------------------------------------------------------------------
    # Plot 3: combined_hnsw_vs_ivf_comparison - 2 rows x 3 cols (large only)
    # Rows: movies, reviews | Cols: FAISS, Milvus, pgvector
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 3, figsize=WORKSHOP_3X2_FIGSIZE)  # 2 rows, 3 cols
    all_metadata = []

    if averages_hnsw is not None and averages_ivf is not None:
        for row_idx, query_type in enumerate(['movies', 'reviews']):
            for col_idx, system_name in enumerate(system_order):
                handles, labels, metadata = plot_hnsw_ivf_comparison(
                    axes3[row_idx, col_idx],
                    averages_hnsw, averages_ivf,
                    DATASET_SIZE,
                    query_type=query_type,
                    system_name=system_name,
                    return_handles_labels=True,
                )
                all_metadata.extend(metadata)

        # Create multi-color legend
        legend_data = {}
        for handle, index_type, sel_idx, sel_value, query_type in all_metadata:
            key = (index_type, sel_idx)
            if key not in legend_data:
                legend_data[key] = {'sel_values': {}, 'linestyle': '--' if sel_idx == 0 else '-'}
            legend_data[key]['sel_values'][query_type] = sel_value

        system_colors_hnsw = {'FAISS': '#00FF00', 'Milvus': '#FF0000', 'pgvector': '#0000FF'}
        system_colors_ivf = {'FAISS': '#66AA66', 'Milvus': '#AA6666', 'pgvector': '#6666AA'}
        markers_sel = ['.', '+', 'x', 's']
        linewidth_multipliers = [0.8, 0.9, 1.4, 2]
        base_linewidth = 2

        legend_handles = []
        legend_labels = []
        sorted_keys = sorted(legend_data.keys(), key=lambda x: (x[0], x[1]))

        for index_type, sel_idx in sorted_keys:
            data = legend_data[(index_type, sel_idx)]
            sel_dict = data['sel_values']
            linestyle = data['linestyle']
            marker = markers_sel[sel_idx] if sel_idx < len(markers_sel) else 'o'
            linewidth = base_linewidth * linewidth_multipliers[sel_idx] if sel_idx < len(linewidth_multipliers) else 2
            sel_strs = []
            for qt, short in [('movies', 'M'), ('reviews', 'R')]:
                if qt in sel_dict:
                    sel_val = sel_dict[qt]
                    sel_str = f'{sel_val:.2f}' if sel_val >= 0.01 else f'{sel_val:.3f}'
                    sel_strs.append(f'{short}: {sel_str}')
            sel_label = '{' + ', '.join(sel_strs) + '}'
            colors = system_colors_hnsw if index_type == 'HNSW' else system_colors_ivf
            handles_tuple = tuple(
                Line2D([0], [0], color=colors[sys_name], marker=marker, linestyle=linestyle,
                       linewidth=linewidth, markersize=10)
                for sys_name in system_order
            )
            legend_handles.append(handles_tuple)
            legend_labels.append(f'{index_type} $\\sigma_g$={sel_label}')

        # Legend below plots: 2 columns (HNSW left, IVFFlat right), selectivity levels as rows.
        # matplotlib fills column-major, and legend_handles is ordered HNSW... then IVF...,
        # so the left column is HNSW and the right column is IVFFlat.
        leg3 = fig3.legend(legend_handles, legend_labels, loc='upper center',
                           bbox_to_anchor=(0.5, 0.25), ncol=2, handlelength=6,
                           fontsize=WORKSHOP_3X2_STYLE['legend'],
                           columnspacing=2.5, handletextpad=0.5, frameon=True,
                           handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)})
    else:
        leg3 = None

    apply_workshop_3x2_style(axes3)
    plt.tight_layout(rect=[0, 0.22, 1, 1])
    fig3.subplots_adjust(wspace=WORKSHOP_3X2_WSPACE)
    if leg3 is not None:
        # Tuck the M/R abbreviation note directly under the legend entries so it reads as part
        # of the (frameless) legend group, while keeping the clean HNSW | IVFFlat columns.
        fig3.canvas.draw()
        _lb = leg3.get_window_extent().transformed(fig3.transFigure.inverted())
        # fig3.text(0.5, _lb.y0 - 0.005, 'M — Movies, R — Reviews', ha='center', va='top', fontsize=20)
    _save3 = dict(dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(plots_dir, 'workshop_combined_hnsw_vs_ivf_comparison.pdf'), **_save3)
    plt.savefig(os.path.join(plots_dir, 'workshop_combined_hnsw_vs_ivf_comparison.png'), **_save3)
    plt.close()
    print(f"Plot 3 saved: {plots_dir}/workshop_combined_hnsw_vs_ivf_comparison.pdf/png")

    # -------------------------------------------------------------------------
    # Plot 4: qps_recall_by_correlation (from create_results_with_correlation.py)
    # 2 rows x 3 cols: rows=[movies, reviews], cols=[FAISS, Milvus, pgvector]
    # -------------------------------------------------------------------------
    if CORRELATION_PLOT_AVAILABLE:
        results_dir = f"{root_results}/MoRe_UPD_{DATASET_SIZE}_attidx_0"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filter_stats_path = os.path.join(project_root, 'data', 'datasets', 'MoRe_large', 'queries', 'filter_stats_0_k2048.csv')
        fid_list = [3, 5, 6]
        fid_str = '_'.join(map(str, fid_list))
        corr_csv_path = os.path.join(results_dir, f'results_with_correlation_fid{fid_str}.csv')

        corr_df = None
        if os.path.exists(corr_csv_path):
            corr_df = pd.read_csv(corr_csv_path)
            print(f"  Loaded correlation data from {corr_csv_path}")
        elif os.path.exists(filter_stats_path):
            try:
                filter_stats = load_filter_stats(filter_stats_path)
                all_corr_results = []
                for fid in fid_list:
                    fid_results = create_results_with_correlation(
                        results_dir, filter_stats, fid, k=10
                    )
                    if len(fid_results) > 0:
                        fid_results = add_recall_from_all_results(fid_results, results_dir, fid)
                        all_corr_results.append(fid_results)
                if all_corr_results:
                    corr_df = pd.concat(all_corr_results, ignore_index=True)
            except Exception as e:
                print(f"  Could not create correlation data: {e}")

        if corr_df is not None and len(corr_df) > 0:
            fig4, axes4 = plt.subplots(2, 3, figsize=WORKSHOP_3X2_FIGSIZE)
            systems = ['FAISS', 'Milvus', 'pgvector']
            query_types = ['movies', 'reviews']
            high_thresh, low_thresh = 0.3, -0.3

            for row, qt in enumerate(query_types):
                for col, system_name in enumerate(systems):
                    plot_qps_recall_by_correlation(
                        corr_df, 'correlation_gls', axes4[row, col],
                        f'{qt.capitalize()}, {system_name}',
                        system=system_name, query_type=qt,
                        high_thresh=high_thresh, low_thresh=low_thresh
                    )

            # Legend on the right, 1 column (matching other plots)
            _sys_hnsw = {'FAISS': '#00FF00', 'Milvus': '#FF0000', 'pgvector': '#0000FF'}
            _sys_ivf = {'FAISS': '#66AA66', 'Milvus': '#AA6666', 'pgvector': '#6666AA'}
            _lw = {'high': 3.5, 'average': 2.0, 'low': 1.0}
            _mk = {'high': 'o', 'average': 's', 'low': '+'}
            _ms = {'high': 6, 'average': 6, 'low': 4}
            legend_handles, legend_labels = [], []
            for idx_type, ls in [('HNSW', '-'), ('IVF', '--')]:
                colors = _sys_hnsw if idx_type == 'HNSW' else _sys_ivf
                for level in ['high', 'average', 'low']:
                    handles = tuple(
                        Line2D([0], [0], color=colors[s], linestyle=ls,
                               linewidth=_lw[level], marker=_mk[level], markersize=_ms[level])
                        for s in systems
                    )
                    legend_handles.append(handles)
                    if level == 'high':
                        legend_labels.append(f'{idx_type}, GLS correlation')
                    elif level == 'average':
                        legend_labels.append(f'{idx_type}, GLS no-correlation')
                    else:
                        legend_labels.append(f'{idx_type}, GLS anti-correlation')

            # Legend below plots: 2 columns x 3 rows (HNSW left, IVFFlat right).
            # matplotlib fills column-major and legend_handles is ordered HNSW... then IVF...,
            # so the left column is HNSW and the right column is IVFFlat.
            apply_workshop_3x2_style(axes4)
            plt.tight_layout(rect=[0, 0.16, 1, 1])
            fig4.subplots_adjust(wspace=WORKSHOP_3X2_WSPACE)
            fig4.legend(legend_handles, legend_labels,
                        handler_map={tuple: HandlerTuple(ndivide=None)},
                        loc='upper center', bbox_to_anchor=(0.5, 0.20), ncol=2,
                        handlelength=5, fontsize=WORKSHOP_3X2_STYLE['legend'],
                        columnspacing=2.5, frameon=True)
            _save4 = dict(dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.savefig(os.path.join(plots_dir, 'workshop_qps_recall_by_correlation.pdf'), **_save4)
            plt.savefig(os.path.join(plots_dir, 'workshop_qps_recall_by_correlation.png'), **_save4)
            plt.close()
            print(f"Plot 4 saved: {plots_dir}/workshop_qps_recall_by_correlation.pdf/png")
        else:
            print("Plot 4 (qps_recall_by_correlation) skipped: no correlation data. Run create_results_with_correlation.py --plot first.")
    else:
        print("Plot 4 (qps_recall_by_correlation) skipped: create_results_with_correlation not available.")


if __name__ == "__main__":
    main()
