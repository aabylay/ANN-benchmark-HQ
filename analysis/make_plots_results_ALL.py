import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os

# Exact cardinalities for movies and reviews datasets
data_sizes = {
    "small": {
        "movies": "small",
        "reviews": "small",
    },
    "medium": {
        "movies": "medium",
        "reviews": "medium",
    },
    "large": {
        "movies": "large",
        "reviews": "large",
    }
}

# Algorithm names for HNSW
algo_names_hnsw = {
    'milvus-hnsw': 'Milvus',
    'pgvector': 'pgvector',
    'hnsw(faiss)': 'FAISS'
}

# Algorithm names for IVF
algo_names_ivf = {
    'faiss-ivf': 'FAISS',
    'milvus-ivfflat': 'Milvus',
    'pgvector_ivf': 'pgvector'
}

# Set matplotlib to use DejaVu Serif (LaTeX-like font) and configure mathtext
mpl.rcParams['lines.linewidth'] = 2
plt.rc('font', family='serif', serif='DejaVu Serif', size=24)
plt.rc('mathtext', default='regular')


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
# HNSW Plot Functions
# =============================================================================

def plot_throughput_vs_recall_by_selectivity_hnsw(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Plot: Throughput vs recall by selectivity for HNSW (k=10, m=10)"""
    markers_sel = ['.', '+', 'x', 's']
    linewidths = [0.6, 1.1, 1.6, 2.2]
    colors_sel = [
        ["#FF6666", "#FF4444", "#FF2222", "#FF0000"],  # milvus-hnsw
        ["#6666FF", "#4444FF", "#2222FF", "#0000FF"],  # pgvector
        ["#66FF66", "#44FF44", "#22FF22", "#00FF00"]   # hnsw(faiss)
    ]
    
    sub_qt = averages[(averages['query_type'] == query_type) & (averages['k'] == 10) & (averages['m'] == 10)]
    unique_sel = sorted(sub_qt['filter_selectivity'].unique())
    unique_sel = [unique_sel[i] for i in [1, 3, 5, -1] if i < len(unique_sel)]
    
    handles = []
    labels = []
    metadata = []
    
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
    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}, @k=10 ({data_sizes[dataset_size][query_type]})')
    ax.set_yscale('log')
    if len(sub_qt) > 0:
        max_throughput = max(sub_qt['throughput'])
        ax.set_yticks([10**i for i in range(1, int(np.ceil(np.log10(max_throughput))))])
    ax.grid(True, which="both", ls="--")
    
    if return_handles_labels:
        return handles, labels, metadata
    return None, None, []


def plot_recall_vs_selectivity_by_k_hnsw(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Plot: Selectivity on x-axis, recall on y-axis, with different k values for HNSW"""
    k_values = sorted(averages['k'].unique())
    
    algo_colors = {
        'milvus-hnsw': ['#FF6666', '#FF4444', '#FF2222', '#FF0000'],
        'pgvector': ['#6666FF', '#4444FF', '#2222FF', '#0000FF'],
        'hnsw(faiss)': ['#66FF66', '#44FF44', '#22FF22', '#00FF00']
    }
    
    k_to_thickness = {k: ((idx + 1) * 0.6 + idx*(idx*0.1)) for idx, k in enumerate(k_values)}
    
    sub_qt = averages[(averages['query_type'] == query_type) & (averages['m'] == 10)]
    
    handles = []
    labels = []
    
    for algo in sub_qt['algorithm'].unique():
        color = algo_colors.get(algo, '#000000')
        for idx, k in enumerate(k_values):
            sub_k = sub_qt[(sub_qt['algorithm'] == algo) & (sub_qt['k'] == k)]
            if len(sub_k) == 0:
                continue
            
            grouped = sub_k.groupby('filter_selectivity').agg({'recall': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='filter_selectivity')
            
            line_thickness = k_to_thickness.get(k, 2.0)
            
            label = f'{algo_names_hnsw[algo]} k={k}'
            line, = ax.plot(grouped['filter_selectivity'], grouped['recall'], 
                   color=color[idx], linewidth=line_thickness, label=label, marker='o', markersize=7)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('Recall')
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0.2, 0.6, 1])
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap} ({data_sizes[dataset_size][query_type]})')
    ax.grid(True, which="both", ls="--")
    
    if return_handles_labels:
        return handles, labels


def plot_throughput_vs_selectivity_by_k_hnsw(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Plot: Selectivity on x-axis, throughput on y-axis, with different k values for HNSW"""
    k_values = sorted(averages['k'].unique())
    
    algo_colors = {
        'milvus-hnsw': ['#FF6666', '#FF4444', '#FF2222', '#FF0000'],
        'pgvector': ['#6666FF', '#4444FF', '#2222FF', '#0000FF'],
        'hnsw(faiss)': ['#66FF66', '#44FF44', '#22FF22', '#00FF00']
    }
    
    k_to_thickness = {k: ((idx + 1) * 0.6 + idx*(idx*0.1)) for idx, k in enumerate(k_values)}
    
    sub_qt = averages[(averages['query_type'] == query_type) & (averages['m'] == 10)]
    
    handles = []
    labels = []
    
    for algo in sub_qt['algorithm'].unique():
        color = algo_colors.get(algo, '#000000')
        for idx, k in enumerate(k_values):
            sub_k = sub_qt[(sub_qt['algorithm'] == algo) & (sub_qt['k'] == k)]
            if len(sub_k) == 0:
                continue
            
            grouped = sub_k.groupby('filter_selectivity').agg({'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='filter_selectivity')
            
            line_thickness = k_to_thickness.get(k, 2.0)
            
            label = f'{algo_names_hnsw[algo]} k={k}'
            line, = ax.plot(grouped['filter_selectivity'], grouped['throughput'], 
                   color=color[idx], linewidth=line_thickness, label=label, marker='o', markersize=7)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
    
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}, @m=10, ({data_sizes[dataset_size][query_type]})')
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    yticks = [10**i for i in range(1, 4)]
    yticks.append(10**3.5)
    ax.set_yticks(yticks)
    ax.grid(True, which="both", ls="--")
    
    if return_handles_labels:
        return handles, labels


# =============================================================================
# IVF Plot Functions
# =============================================================================

def plot_throughput_vs_recall_by_selectivity_ivf(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Plot: Throughput vs recall by selectivity for IVF (k=10)"""
    markers_sel = ['.', '+', 'x', 's']
    linewidths = [0.6, 1.1, 1.6, 2.2]
    colors_sel = [
        ["#FF6666", "#FF4444", "#FF2222", "#FF0000"],  # milvus-ivf
        ["#6666FF", "#4444FF", "#2222FF", "#0000FF"],  # pgvector
        ["#66FF66", "#44FF44", "#22FF22", "#00FF00"]   # ivf(faiss)
    ]
    
    sub_qt = averages[(averages['query_type'] == query_type) & (averages['k'] == 10)]
    
    unique_sel = sorted(sub_qt['filter_selectivity'].unique())
    unique_sel = [unique_sel[i] for i in [1, 3, 5, -1] if i < len(unique_sel)]
    
    handles = []
    labels = []
    metadata = []
    
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
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}, @k=10 ({data_sizes[dataset_size][query_type]})')
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**0.2, 10**3.5])
    ax.set_yticks([10**i for i in range(1, 4)])
    ax.grid(True, which="both", ls="--")
    
    if return_handles_labels:
        return handles, labels, metadata
    return None, None, []


def plot_recall_vs_selectivity_by_k_ivf(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Plot: Selectivity on x-axis, recall on y-axis, with different k values for IVF"""
    k_values = sorted(averages['k'].unique())
    
    algo_colors = {
        'milvus-ivfflat': ["#FF6666", "#FF4444", "#FF2222", "#FF0000"],
        'pgvector_ivf': ["#6666FF", "#4444FF", "#2222FF", "#0000FF"],
        'faiss-ivf': ["#66FF66", "#44FF44", "#22FF22", "#00FF00"]
    }
    
    k_to_thickness = {k: ((idx + 1) * 0.6 + idx*(idx*0.1)) for idx, k in enumerate(k_values)}
    
    sub_qt = averages[averages['query_type'] == query_type]
    
    handles = []
    labels = []
    
    for algo in sub_qt['algorithm'].unique():
        color = algo_colors.get(algo, '#000000')
        for idx, k in enumerate(k_values):
            sub_k = sub_qt[(sub_qt['algorithm'] == algo) & (sub_qt['k'] == k)]
            if len(sub_k) == 0:
                continue
            
            grouped = sub_k.groupby('filter_selectivity').agg({'recall': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='filter_selectivity')
            
            line_thickness = k_to_thickness.get(k, 2.0)
            
            label = f'{algo_names_ivf[algo]} k={k}'
            line, = ax.plot(grouped['filter_selectivity'], grouped['recall'], 
                   color=color[idx], linewidth=line_thickness, label=label, marker='o', markersize=7)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
    
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('Recall')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}, ({data_sizes[dataset_size][query_type]})')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0.2, 0.6, 1])
    ax.grid(True, which="both", ls="--")
    
    if return_handles_labels:
        return handles, labels


def plot_throughput_vs_selectivity_by_k_ivf(ax, averages, dataset_size, query_type='movies', return_handles_labels=False):
    """Plot: Selectivity on x-axis, throughput on y-axis, with different k values for IVF"""
    k_values = sorted(averages['k'].unique())
    
    algo_colors = {
        'milvus-ivfflat': ["#FF6666", "#FF4444", "#FF2222", "#FF0000"],
        'pgvector_ivf': ["#6666FF", "#4444FF", "#2222FF", "#0000FF"],
        'faiss-ivf': ["#66FF66", "#44FF44", "#22FF22", "#00FF00"]
    }
    
    k_to_thickness = {k: ((idx + 1) * 0.6 + idx*(idx*0.1)) for idx, k in enumerate(k_values)}
    
    sub_qt = averages[averages['query_type'] == query_type]
    
    handles = []
    labels = []
    
    for algo in sub_qt['algorithm'].unique():
        color = algo_colors.get(algo, '#000000')
        for idx, k in enumerate(k_values):
            sub_k = sub_qt[(sub_qt['algorithm'] == algo) & (sub_qt['k'] == k)]
            if len(sub_k) == 0:
                continue
            
            grouped = sub_k.groupby('filter_selectivity').agg({'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='filter_selectivity')
            
            line_thickness = k_to_thickness.get(k, 2.0)
            
            label = f'{algo_names_ivf[algo]} k={k}'
            line, = ax.plot(grouped['filter_selectivity'], grouped['throughput'], 
                   color=color[idx], linewidth=line_thickness, label=label, marker='o', markersize=7)
            if return_handles_labels:
                handles.append(line)
                labels.append(label)
    
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap}, ({data_sizes[dataset_size][query_type]})')
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    yticks = [10**i for i in range(1, 4)]
    yticks.append(10**3.5)
    ax.set_yticks(yticks)
    ax.grid(True, which="both", ls="--")
    
    if return_handles_labels:
        return handles, labels


def create_unified_legend_throughput_recall(all_metadata_hnsw, all_metadata_ivf, algo_names):
    """Create unified legend for throughput vs recall by selectivity plot"""
    legend_data = {}
    
    # Process HNSW metadata
    for handle, algo, sel_idx, sel_value, dataset_size in all_metadata_hnsw:
        key = (algo, sel_idx, 'hnsw')
        if key not in legend_data:
            legend_data[key] = {'handle': handle, 'sel_values': {}, 'algo_name': algo_names.get(algo, algo)}
        legend_data[key]['sel_values'][dataset_size] = sel_value
    
    # Process IVF metadata
    for handle, algo, sel_idx, sel_value, dataset_size in all_metadata_ivf:
        key = (algo, sel_idx, 'ivf')
        if key not in legend_data:
            legend_data[key] = {'handle': handle, 'sel_values': {}, 'algo_name': algo_names.get(algo, algo)}
        legend_data[key]['sel_values'][dataset_size] = sel_value
    
    legend_handles = []
    legend_labels = []
    
    # Sort by algorithm name, then by selectivity index
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


def main():
    root_results = '/home/abylay/ann-benchmarks-HQ/results'
    dataset_sizes = ['small', 'medium', 'large']
    plots_dir = f"{root_results}/MoRe_UPD_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load HNSW data for all dataset sizes
    averages_hnsw = {}
    for dataset_size in dataset_sizes:
        csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_0/all_results_hnsw.csv"
        try:
            df = pd.read_csv(csv_path)
            averages_hnsw[dataset_size] = compute_averages_hnsw(df)
        except FileNotFoundError:
            print(f"Error: HNSW CSV file not found at {csv_path}")
            averages_hnsw[dataset_size] = None
    
    # Load IVF data for all dataset sizes
    averages_ivf = {}
    for dataset_size in dataset_sizes:
        csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_0/all_results_ivf.csv"
        try:
            df = pd.read_csv(csv_path)
            averages_ivf[dataset_size] = compute_averages_ivf(df)
        except FileNotFoundError:
            print(f"Error: IVF CSV file not found at {csv_path}")
            averages_ivf[dataset_size] = None
    
    # =========================================================================
    # Combined Plot 1: Throughput vs Recall by Selectivity
    # =========================================================================
    # Use GridSpec with 5 rows: 2 for HNSW, 1 for gap/title, 2 for IVF
    fig1 = plt.figure(figsize=(24, 18.5))
    gs1 = GridSpec(5, 3, figure=fig1, height_ratios=[1, 1, -0.12, 1, 1], hspace=0.63, wspace=0.25, top=0.93, bottom=0.08)
    
    # Create axes array mapping: logical rows 0-3 to GridSpec rows 0,1,3,4
    axes1 = np.empty((4, 3), dtype=object)
    gs_row_map = {0: 0, 1: 1, 2: 3, 3: 4}  # Skip row 2 (gap)
    for row in range(4):
        for col in range(3):
            axes1[row, col] = fig1.add_subplot(gs1[gs_row_map[row], col])
    
    all_metadata_hnsw = []
    all_metadata_ivf = []
    
    # HNSW plots (rows 0-1)
    for col_idx, dataset_size in enumerate(dataset_sizes):
        for row_idx, query_type in enumerate(['movies', 'reviews']):
            if averages_hnsw[dataset_size] is not None:
                handles, labels, metadata = plot_throughput_vs_recall_by_selectivity_hnsw(
                    axes1[row_idx, col_idx], averages_hnsw[dataset_size], dataset_size, query_type, return_handles_labels=True)
                if row_idx == 0:
                    all_metadata_hnsw.extend(metadata)
    
    # IVF plots (rows 2-3)
    for col_idx, dataset_size in enumerate(dataset_sizes):
        for row_idx, query_type in enumerate(['movies', 'reviews']):
            if averages_ivf[dataset_size] is not None:
                handles, labels, metadata = plot_throughput_vs_recall_by_selectivity_ivf(
                    axes1[row_idx + 2, col_idx], averages_ivf[dataset_size], dataset_size, query_type, return_handles_labels=True)
                if row_idx == 0:
                    all_metadata_ivf.extend(metadata)
    
    # Add section titles (positioned in the gaps)
    fig1.text(0.5, 0.96, '(a) HNSW', ha='center', va='bottom', fontsize=28, fontweight='bold')
    fig1.text(0.5, 0.48, '(b) IVFFlat', ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    # Create unified legend (use HNSW metadata only since they share same system names)
    legend_handles, legend_labels = create_unified_legend_throughput_recall(
        all_metadata_hnsw, [], algo_names_hnsw)
    
    n_entries = len(legend_handles)
    print("n_entries:", n_entries)
    fig1.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                ncol=min(n_entries, 3))
    
    plt.savefig(os.path.join(plots_dir, 'combined_throughput_vs_recall_by_selectivity.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'combined_throughput_vs_recall_by_selectivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined Plot 1 saved in: {plots_dir}/combined_throughput_vs_recall_by_selectivity.pdf/png")
    
    # =========================================================================
    # Combined Plot 2: Recall vs Selectivity by k
    # =========================================================================
    # Use GridSpec with 5 rows: 2 for HNSW, 1 for gap/title, 2 for IVF
    fig2 = plt.figure(figsize=(24, 18.5))
    gs2 = GridSpec(5, 3, figure=fig2, height_ratios=[1, 1, -0.1, 1, 1], hspace=0.55, wspace=0.25, top=0.93, bottom=0.08)
    
    # Create axes array mapping: logical rows 0-3 to GridSpec rows 0,1,3,4
    axes2 = np.empty((4, 3), dtype=object)
    gs_row_map = {0: 0, 1: 1, 2: 3, 3: 4}  # Skip row 2 (gap)
    for row in range(4):
        for col in range(3):
            axes2[row, col] = fig2.add_subplot(gs2[gs_row_map[row], col])
    
    all_handles_hnsw, all_labels_hnsw = [], []
    all_handles_ivf, all_labels_ivf = [], []
    
    # HNSW plots (rows 0-1)
    for row_idx, query_type in enumerate(['movies', 'reviews']):
        for col_idx, dataset_size in enumerate(dataset_sizes):
            if averages_hnsw[dataset_size] is not None:
                handles, labels = plot_recall_vs_selectivity_by_k_hnsw(
                    axes2[row_idx, col_idx], averages_hnsw[dataset_size], dataset_size, query_type, return_handles_labels=True)
                if row_idx == 0 and col_idx == 0:
                    all_handles_hnsw.extend(handles)
                    all_labels_hnsw.extend(labels)
    
    # IVF plots (rows 2-3)
    for row_idx, query_type in enumerate(['movies', 'reviews']):
        for col_idx, dataset_size in enumerate(dataset_sizes):
            if averages_ivf[dataset_size] is not None:
                handles, labels = plot_recall_vs_selectivity_by_k_ivf(
                    axes2[row_idx + 2, col_idx], averages_ivf[dataset_size], dataset_size, query_type, return_handles_labels=True)
                if row_idx == 0 and col_idx == 0:
                    all_handles_ivf.extend(handles)
                    all_labels_ivf.extend(labels)
    
    # Add section titles (positioned in the gaps)
    fig2.text(0.5, 0.96, '(a) HNSW', ha='center', va='bottom', fontsize=28, fontweight='bold')
    fig2.text(0.5, 0.48, '(b) IVFFlat', ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    # Group by algorithm for multi-row legend (use HNSW handles since they share same system names)
    algo_groups = {}
    for handle, label in zip(all_handles_hnsw, all_labels_hnsw):
        algo_name = label.split()[0]
        if algo_name not in algo_groups:
            algo_groups[algo_name] = []
        algo_groups[algo_name].append((handle, label))
    
    sorted_handles = []
    sorted_labels = []
    for algo in sorted(algo_groups.keys()):
        for handle, label in algo_groups[algo]:
            sorted_handles.append(handle)
            sorted_labels.append(label)
    
    n_per_row = len(algo_groups[sorted(algo_groups.keys())[0]]) if algo_groups else 4
    #print("algo:", sorted(algo_groups.keys())[0])
    #print("values:", algo_groups[sorted(algo_groups.keys())[0]])
    #print("n_per_row:", n_per_row)
    fig2.legend(sorted_handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                ncol=max(n_per_row - 1, 1))
    
    plt.savefig(os.path.join(plots_dir, 'combined_recall_vs_selectivity_by_k.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'combined_recall_vs_selectivity_by_k.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined Plot 2 saved in: {plots_dir}/combined_recall_vs_selectivity_by_k.pdf/png")
    
    # =========================================================================
    # Combined Plot 3: Throughput vs Selectivity by k
    # =========================================================================
    # Use GridSpec with 5 rows: 2 for HNSW, 1 for gap/title, 2 for IVF
    fig3 = plt.figure(figsize=(24, 18.5))
    gs3 = GridSpec(5, 3, figure=fig3, height_ratios=[1, 1, -0.1, 1, 1], hspace=0.55, wspace=0.25, top=0.93, bottom=0.08)
    
    # Create axes array mapping: logical rows 0-3 to GridSpec rows 0,1,3,4
    axes3 = np.empty((4, 3), dtype=object)
    gs_row_map = {0: 0, 1: 1, 2: 3, 3: 4}  # Skip row 2 (gap)
    for row in range(4):
        for col in range(3):
            axes3[row, col] = fig3.add_subplot(gs3[gs_row_map[row], col])
    
    all_handles_hnsw, all_labels_hnsw = [], []
    all_handles_ivf, all_labels_ivf = [], []
    
    # HNSW plots (rows 0-1)
    for row_idx, query_type in enumerate(['movies', 'reviews']):
        for col_idx, dataset_size in enumerate(dataset_sizes):
            if averages_hnsw[dataset_size] is not None:
                handles, labels = plot_throughput_vs_selectivity_by_k_hnsw(
                    axes3[row_idx, col_idx], averages_hnsw[dataset_size], dataset_size, query_type, return_handles_labels=True)
                if row_idx == 0 and col_idx == 0:
                    all_handles_hnsw.extend(handles)
                    all_labels_hnsw.extend(labels)
    
    # IVF plots (rows 2-3)
    for row_idx, query_type in enumerate(['movies', 'reviews']):
        for col_idx, dataset_size in enumerate(dataset_sizes):
            if averages_ivf[dataset_size] is not None:
                handles, labels = plot_throughput_vs_selectivity_by_k_ivf(
                    axes3[row_idx + 2, col_idx], averages_ivf[dataset_size], dataset_size, query_type, return_handles_labels=True)
                if row_idx == 0 and col_idx == 0:
                    all_handles_ivf.extend(handles)
                    all_labels_ivf.extend(labels)
    
    # Add section titles (positioned in the gaps)
    fig3.text(0.5, 0.96, '(a) HNSW', ha='center', va='bottom', fontsize=28, fontweight='bold')
    fig3.text(0.5, 0.48, '(b) IVFFlat', ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    # Group by algorithm for multi-row legend (use HNSW handles since they share same system names)
    algo_groups = {}
    for handle, label in zip(all_handles_hnsw, all_labels_hnsw):
        algo_name = label.split()[0]
        if algo_name not in algo_groups:
            algo_groups[algo_name] = []
        algo_groups[algo_name].append((handle, label))
    
    sorted_handles = []
    sorted_labels = []
    for algo in sorted(algo_groups.keys()):
        for handle, label in algo_groups[algo]:
            sorted_handles.append(handle)
            sorted_labels.append(label)
    
    n_per_row = len(algo_groups[sorted(algo_groups.keys())[0]]) if algo_groups else 4
    fig3.legend(sorted_handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                ncol=max(n_per_row - 1, 1))
    
    plt.savefig(os.path.join(plots_dir, 'combined_throughput_vs_selectivity_by_k.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'combined_throughput_vs_selectivity_by_k.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined Plot 3 saved in: {plots_dir}/combined_throughput_vs_selectivity_by_k.pdf/png")


if __name__ == "__main__":
    main()
