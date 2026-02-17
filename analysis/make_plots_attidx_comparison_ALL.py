import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.lines import Line2D
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


def plot_throughput_vs_recall_attidx_hnsw(ax, averages_0, averages_1, dataset_size, query_type='movies', algorithm='pgvector', return_handles_labels=False):
    """Plot: Throughput vs recall by selectivity comparing att_idx=0 vs att_idx=1 for HNSW (k=10, m=10)"""
    markers_sel = ['D', 'X', 'o']
    # Colors based on system: pgvector=blue, milvus=red
    algo_display = 'Milvus' if algorithm == 'milvus-hnsw' else 'pgvector'
    if algo_display == 'pgvector':
        system_color = "#0000FF"  # Blue for pgvector
        system_color2 = "#8888FF"
    else:
        system_color = "#FF0000"  # Red for Milvus
        system_color2 = "#FF8888"
    base_linewidth = 2
    
    # Handle None values
    if averages_0 is None:
        averages_0 = pd.DataFrame()
    if averages_1 is None:
        averages_1 = pd.DataFrame()
    
    # Filter for specific query type, algorithm, and parameters
    if len(averages_0) > 0:
        sub_qt_0 = averages_0[(averages_0['query_type'] == query_type) & 
                              (averages_0['k'] == 10) & 
                              (averages_0['m'] == 10) &
                              (averages_0['algorithm'] == algorithm)]
    else:
        sub_qt_0 = pd.DataFrame()
    
    if len(averages_1) > 0:
        sub_qt_1 = averages_1[(averages_1['query_type'] == query_type) & 
                              (averages_1['k'] == 10) & 
                              (averages_1['m'] == 10) &
                              (averages_1['algorithm'] == algorithm)]
    else:
        sub_qt_1 = pd.DataFrame()
    
    # Get common selectivities
    unique_sel_0 = sorted(sub_qt_0['filter_selectivity'].unique()) if len(sub_qt_0) > 0 else []
    unique_sel_1 = sorted(sub_qt_1['filter_selectivity'].unique()) if len(sub_qt_1) > 0 else []
    unique_sel = sorted(set(unique_sel_0) & set(unique_sel_1)) if (unique_sel_0 and unique_sel_1) else sorted(set(unique_sel_0) | set(unique_sel_1))
    unique_sel = [unique_sel[i] for i in [1, 4, -1] if i < len(unique_sel)]
    
    handles = []
    labels = []
    metadata = []  # Store (handle, att_idx, sel_idx, sel_value, query_type)
    
    # Calculate linewidth multipliers
    n_sel = len(unique_sel[:len(markers_sel)])
    linewidth_multipliers = []
    for i in range(n_sel):
        if n_sel == 1:
            mult = 2.0
        elif i == n_sel - 1:
            mult = 2.0
        elif i == 0:
            mult = 0.6
        else:
            mult = 1.3
        linewidth_multipliers.append(mult)
    
    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel_0 = sub_qt_0[sub_qt_0['filter_selectivity'] == sel] if len(sub_qt_0) > 0 else pd.DataFrame()
        sub_sel_1 = sub_qt_1[sub_qt_1['filter_selectivity'] == sel] if len(sub_qt_1) > 0 else pd.DataFrame()
        
        linewidth = base_linewidth * linewidth_multipliers[idx]
        
        # Plot att_idx=0 (solid line, system color)
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
        
        # Plot att_idx=1 (dashed line, lighter system color)
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
    ax.set_xlabel('Recall@k=10')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap} ({data_sizes[dataset_size][query_type]})')
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    if algo_display == 'Milvus':
        ax.set_xlim([0.8, 1.01])
        ax.set_ylim([30, 260])
        yticks = [2**5, 2**6, 2**7, 2**8]  # 32, 64, 128, 256
        ax.set_yticks(yticks)
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_minor_locator(NullLocator())  # Remove minor ticks
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'$2^{{{int(np.log2(x))}}}$'))
    elif algo_display == 'pgvector':
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
    
    if return_handles_labels:
        return handles, labels, metadata
    return None, None, []


def plot_throughput_vs_recall_attidx_ivf(ax, averages_0, averages_1, dataset_size, query_type='movies', algorithm='pgvector_ivf', return_handles_labels=False):
    """Plot: Throughput vs recall by selectivity comparing att_idx=0 vs att_idx=1 for IVF (k=10)"""
    markers_sel = ['D', 'X', 'o']
    # Colors based on system: pgvector=blue, milvus=red
    algo_display = 'Milvus' if algorithm == 'milvus-ivfflat' else 'pgvector'
    if algo_display == 'pgvector':
        system_color = "#0000FF"  # Blue for pgvector
        system_color2 = "#8888FF"
    else:
        system_color = "#FF0000"  # Red for Milvus
        system_color2 = "#FF8888"
    base_linewidth = 2
    
    # Handle None values
    if averages_0 is None:
        averages_0 = pd.DataFrame()
    if averages_1 is None:
        averages_1 = pd.DataFrame()
    
    # Filter for specific query type, algorithm, and k=10
    if len(averages_0) > 0:
        sub_qt_0 = averages_0[(averages_0['query_type'] == query_type) & 
                              (averages_0['k'] == 10) &
                              (averages_0['algorithm'] == algorithm)]
    else:
        sub_qt_0 = pd.DataFrame()
    
    if len(averages_1) > 0:
        sub_qt_1 = averages_1[(averages_1['query_type'] == query_type) & 
                              (averages_1['k'] == 10) &
                              (averages_1['algorithm'] == algorithm)]
    else:
        sub_qt_1 = pd.DataFrame()
    
    # Get common selectivities
    unique_sel_0 = sorted(sub_qt_0['filter_selectivity'].unique()) if len(sub_qt_0) > 0 else []
    unique_sel_1 = sorted(sub_qt_1['filter_selectivity'].unique()) if len(sub_qt_1) > 0 else []
    unique_sel = sorted(set(unique_sel_0) & set(unique_sel_1)) if (unique_sel_0 and unique_sel_1) else sorted(set(unique_sel_0) | set(unique_sel_1))
    unique_sel = [unique_sel[i] for i in [1, 4, -1] if i < len(unique_sel)]
    
    handles = []
    labels = []
    metadata = []
    
    # Calculate linewidth multipliers
    n_sel = len(unique_sel[:len(markers_sel)])
    linewidth_multipliers = []
    for i in range(n_sel):
        if n_sel == 1:
            mult = 2.0
        elif i == n_sel - 1:
            mult = 2.0
        elif i == 0:
            mult = 0.6
        else:
            mult = 1.3
        linewidth_multipliers.append(mult)
    
    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel_0 = sub_qt_0[sub_qt_0['filter_selectivity'] == sel] if len(sub_qt_0) > 0 else pd.DataFrame()
        sub_sel_1 = sub_qt_1[sub_qt_1['filter_selectivity'] == sel] if len(sub_qt_1) > 0 else pd.DataFrame()
        
        linewidth = base_linewidth * linewidth_multipliers[idx]
        
        # Plot att_idx=0 (solid line, system color)
        if len(sub_sel_0) > 0:
            grouped_0 = sub_sel_0.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped_0 = grouped_0.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label_0 = f'att_idx=0, $\\sigma_g$≈{sel_approx}'
            line_0, = ax.plot(grouped_0['recall'], grouped_0['throughput'], 
                             marker=markers_sel[idx], markersize=10, color=system_color, 
                             linestyle='-', label=label_0, linewidth=linewidth)
            if return_handles_labels:
                handles.append(line_0)
                labels.append(label_0)
                metadata.append((line_0, 0, idx, sel, query_type))
        
        # Plot att_idx=1 (dashed line, lighter system color)
        if len(sub_sel_1) > 0:
            grouped_1 = sub_sel_1.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped_1 = grouped_1.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label_1 = f'att_idx=1, $\\sigma_g$≈{sel_approx}'
            line_1, = ax.plot(grouped_1['recall'], grouped_1['throughput'], 
                             marker=markers_sel[idx], markersize=14, color=system_color2, 
                             linestyle='--', label=label_1, linewidth=linewidth)
            if return_handles_labels:
                handles.append(line_1)
                labels.append(label_1)
                metadata.append((line_1, 1, idx, sel, query_type))
    
    ax.set_xlabel('Recall@k=10')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    ax.set_title(f'{query_type_cap} ({data_sizes[dataset_size][query_type]})')
    ax.axvline(x=1.0, color='#000000', linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    if algo_display == 'Milvus':
        ax.set_xlim([0.7, 1.015])
        ax.set_ylim([10**1, 10**2.5])
        yticks = [10**i for i in range(1, 3)]
        ax.set_yticks(yticks)
    elif algo_display == 'pgvector':
        ax.set_xlim([0, 1.05])
        ax.set_ylim([10**0, 10**3.5])
        yticks = [10**i for i in range(0, 4)]
        ax.set_yticks(yticks)
    ax.grid(True, which="both", ls="--", zorder=0)
    
    if return_handles_labels:
        return handles, labels, metadata
    return None, None, []


def main():
    root_results = '/home/abylay/ann-benchmarks-HQ/results'
    dataset_sizes = ['small', 'medium', 'large']
    plots_dir = f"{root_results}/MoRe_UPD_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load HNSW data for all dataset sizes and both att_idx values
    hnsw_averages_dict = {}
    for dataset_size in dataset_sizes:
        hnsw_averages_dict[dataset_size] = {}
        for att_idx in [0, 1]:
            csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_{att_idx}/all_results_hnsw.csv"
            try:
                df = pd.read_csv(csv_path)
                hnsw_averages_dict[dataset_size][att_idx] = compute_averages_hnsw(df)
            except FileNotFoundError:
                print(f"Error: HNSW CSV file not found at {csv_path}")
                hnsw_averages_dict[dataset_size][att_idx] = None
    
    # Load IVF data for all dataset sizes and both att_idx values
    ivf_averages_dict = {}
    for dataset_size in dataset_sizes:
        ivf_averages_dict[dataset_size] = {}
        for att_idx in [0, 1]:
            csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_{att_idx}/all_results_ivf.csv"
            try:
                df = pd.read_csv(csv_path)
                ivf_averages_dict[dataset_size][att_idx] = compute_averages_ivf(df)
            except FileNotFoundError:
                print(f"Error: IVF CSV file not found at {csv_path}")
                ivf_averages_dict[dataset_size][att_idx] = None
    
    # System configurations: (hnsw_algo, ivf_algo, display_name)
    systems = [
        ('pgvector', 'pgvector_ivf', 'pgvector'),
        ('milvus-hnsw', 'milvus-ivfflat', 'Milvus')
    ]
    
    for hnsw_algo, ivf_algo, algo_display in systems:
        # Create combined figure with HNSW and IVF sections
        # GridSpec with 5 rows: 2 for HNSW, 1 for gap/title, 2 for IVF
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(5, 3, figure=fig, 
                      height_ratios=[1, 1, -0.12, 1, 1], 
                      hspace=0.63, wspace=0.25, top=0.93, bottom=0.08)
        
        # Create axes array: 4 logical rows x 3 columns
        axes = np.empty((4, 3), dtype=object)
        gs_row_map = {0: 0, 1: 1, 2: 3, 3: 4}  # Skip row 2 (gap)
        for row in range(4):
            for col in range(3):
                axes[row, col] = fig.add_subplot(gs[gs_row_map[row], col])
        
        all_metadata = []
        
        # HNSW plots (rows 0-1)
        for row_idx, query_type in enumerate(['movies', 'reviews']):
            for col_idx, dataset_size in enumerate(dataset_sizes):
                if hnsw_averages_dict[dataset_size][0] is not None or hnsw_averages_dict[dataset_size][1] is not None:
                    handles, labels, metadata = plot_throughput_vs_recall_attidx_hnsw(
                        axes[row_idx, col_idx], 
                        hnsw_averages_dict[dataset_size][0],
                        hnsw_averages_dict[dataset_size][1],
                        dataset_size, query_type, hnsw_algo, 
                        return_handles_labels=True
                    )
                    # Collect metadata from first row only
                    if row_idx == 0:
                        all_metadata.extend(metadata)
        
        # IVF plots (rows 2-3)
        for row_idx, query_type in enumerate(['movies', 'reviews']):
            for col_idx, dataset_size in enumerate(dataset_sizes):
                if ivf_averages_dict[dataset_size][0] is not None or ivf_averages_dict[dataset_size][1] is not None:
                    handles, labels, metadata = plot_throughput_vs_recall_attidx_ivf(
                        axes[row_idx + 2, col_idx], 
                        ivf_averages_dict[dataset_size][0],
                        ivf_averages_dict[dataset_size][1],
                        dataset_size, query_type, ivf_algo, 
                        return_handles_labels=True
                    )
        
        # Add section titles
        fig.text(0.5, 0.96, '(a) HNSW', ha='center', va='bottom', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.48, '(b) IVFFlat', ha='center', va='bottom', fontsize=28, fontweight='bold')
        
        # Create unified legend using tuple handles for both att_idx colors
        legend_data = {}
        for handle, att_idx, sel_idx, sel_value, query_type in all_metadata:
            key = (att_idx, sel_idx)
            if key not in legend_data:
                legend_data[key] = {'sel_values': {}, 'handle': handle}
            legend_data[key]['sel_values'][query_type] = sel_value
        
        # System colors
        if algo_display == 'pgvector':
            color_att0 = "#0000FF"
            color_att1 = "#8888FF"
        else:
            color_att0 = "#FF0000"
            color_att1 = "#FF8888"
        
        # Markers and linewidths
        markers_sel = ['D', 'X', 'o']
        base_linewidth = 2
        
        legend_handles = []
        legend_labels = []
        
        # Sort by att_idx (0 first, then 1), then by selectivity index
        sorted_keys = sorted(legend_data.keys(), key=lambda x: (x[0], x[1]))
        
        for att_idx, sel_idx in sorted_keys:
            data = legend_data[(att_idx, sel_idx)]
            sel_dict = data['sel_values']
            marker = markers_sel[sel_idx] if sel_idx < len(markers_sel) else 'o'
            
            # Calculate linewidth
            n_sel = len(set(sk[1] for sk in sorted_keys if sk[0] == att_idx))
            if n_sel == 1:
                mult = 2.0
            elif sel_idx == n_sel - 1:
                mult = 2.0
            elif sel_idx == 0:
                mult = 0.6
            else:
                mult = 1.3
            linewidth = base_linewidth * mult
            
            # Format selectivity label
            sel_strs = []
            for qt in ['movies', 'reviews']:
                if qt in sel_dict:
                    sel_val = sel_dict[qt]
                    sel_str = f'{sel_val:.2f}' if sel_val >= 0.01 else f'{sel_val:.3f}'
                    sel_strs.append(f'{qt}: {sel_str}')
            sel_label = '{' + ', '.join(sel_strs) + '}'
            
            # Create handle with appropriate color and style
            color = color_att0 if att_idx == 0 else color_att1
            linestyle = '-' if att_idx == 0 else '--'
            markersize = 10 if att_idx == 0 else 14
            
            handle = Line2D([0], [0], color=color, marker=marker, linestyle=linestyle, 
                           linewidth=linewidth, markersize=markersize)
            legend_handles.append(handle)
            legend_labels.append(f'att_idx={att_idx}, $\\sigma_g$={sel_label}')
        
        # Place legend at the bottom center
        fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                   ncol=2, fontsize=20, handlelength=4)
        
        plt.savefig(os.path.join(plots_dir, f'combined_attidx_comparison_{algo_display.lower()}.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, f'combined_attidx_comparison_{algo_display.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined plot saved: {plots_dir}/combined_attidx_comparison_{algo_display.lower()}.pdf/png")


if __name__ == "__main__":
    main()
