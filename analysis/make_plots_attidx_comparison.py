import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
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

def compute_averages(df):
    df['query_type'] = df['query_id'].apply(lambda x: 'movies' if x.startswith('qm') else 'reviews')
    averages = df.groupby(['query_type', 'filter_id', 'filter_selectivity', 'k', 'ef_search', 'algorithm', 'm'])[['recall', 'runtime']].mean().reset_index()
    averages['throughput'] = 1 / averages['runtime']
    return averages

def plot_throughput_vs_recall_by_selectivity_attidx(ax, averages_0, averages_1, dataset_size, query_type='movies', algorithm='pgvector', return_handles_labels=False):
    """Plot: Throughput vs recall by selectivity comparing att_idx=0 vs att_idx=1 (k=10, m=10)"""
    markers_sel = ['D', 'X', 'o']
    # Colors based on system: pgvector=blue, milvus=red
    algo_display = 'Milvus' if algorithm == 'milvus-hnsw' else 'pgvector'
    if algo_display == 'pgvector':
        system_color = "#0000FF"  # Blue for pgvector
        system_color2 = "#8888FF"
    else:
        system_color = "#FF0000"  # Red for Milvus
        system_color2 = "#FF8888"
    # Use same color for all selectivities, vary by linewidth
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
        # Verify filtering: ensure all rows have k=10 and m=10
        if len(sub_qt_0) > 0:
            assert all(sub_qt_0['k'] == 10), f"Verification failed: Found k != 10 in sub_qt_0 for {query_type}, {algorithm}"
            assert all(sub_qt_0['m'] == 10), f"Verification failed: Found m != 10 in sub_qt_0 for {query_type}, {algorithm}"
    else:
        sub_qt_0 = pd.DataFrame()
    
    if len(averages_1) > 0:
        sub_qt_1 = averages_1[(averages_1['query_type'] == query_type) & 
                              (averages_1['k'] == 10) & 
                              (averages_1['m'] == 10) &
                              (averages_1['algorithm'] == algorithm)]
        # Verify filtering: ensure all rows have k=10 and m=10
        if len(sub_qt_1) > 0:
            assert all(sub_qt_1['k'] == 10), f"Verification failed: Found k != 10 in sub_qt_1 for {query_type}, {algorithm}"
            assert all(sub_qt_1['m'] == 10), f"Verification failed: Found m != 10 in sub_qt_1 for {query_type}, {algorithm}"
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
    
    # Calculate linewidth multipliers: highest selectivity (last) = x2, middle = x1.5, lowest = x1
    n_sel = len(unique_sel[:len(markers_sel)])
    linewidth_multipliers, line_styles = [], []
    for i in range(n_sel):
        if n_sel == 1:
            mult = 2.0  # Only one selectivity, use highest
            linestyle = '-'
        elif i == n_sel - 1:
            mult = 2.0  # Highest selectivity
            linestyle = '--'
        elif i == 0:
            mult = 0.6  # Lowest selectivity
            linestyle = '--'
        else:
            mult = 1.3  # Middle selectivities
            linestyle = '--'
        linewidth_multipliers.append(mult)
        line_styles.append(linestyle)
    
    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel_0 = sub_qt_0[sub_qt_0['filter_selectivity'] == sel] if len(sub_qt_0) > 0 else pd.DataFrame()
        sub_sel_1 = sub_qt_1[sub_qt_1['filter_selectivity'] == sel] if len(sub_qt_1) > 0 else pd.DataFrame()
        
        # Calculate linewidth for this selectivity
        linewidth = base_linewidth * linewidth_multipliers[idx]
        linestyle = line_styles[idx]
        
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
        
        # Plot att_idx=1 (dotted line, lighter system color)
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
                
        print(f"QPS for att_idx=0 at selectivity {sel}: {grouped_0['throughput'].mean()}")
        print(f"Recall for att_idx=0 at selectivity {sel}: {grouped_0['recall'].mean()}")
        print(f"QPS for att_idx=1 at selectivity {sel}: {grouped_1['throughput'].mean()}")
        print(f"Recall for att_idx=1 at selectivity {sel}: {grouped_1['recall'].mean()}")
        print("--------------------------------")
    
    ax.axvline(x=1.0, color='#000000', linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    query_type_cap = query_type.capitalize()
    algo_display = 'Milvus' if algorithm == 'milvus-hnsw' else 'pgvector'
    ax.set_title(f'{query_type_cap}, @k=10, @m=10, ({data_sizes[dataset_size][query_type]})')
    ax.set_yscale('log')
    ax.set_xlim([0, 1])
    ax.set_ylim([10**1, 10**3.5])
    if algo_display == 'Milvus':
        ax.set_xlim([0.8, 1.01])
        # Set y-axis limits to include custom ticks: 50, 100, 200
        ax.set_ylim([41, 250])
    elif algo_display == 'pgvector':
        ax.set_xlim([0, 1.05])
    if len(sub_qt_0) > 0 or len(sub_qt_1) > 0:
        max_throughput = max(
            max(sub_qt_0['throughput']) if len(sub_qt_0) > 0 else 0,
            max(sub_qt_1['throughput']) if len(sub_qt_1) > 0 else 0
        )
        yticks = [10**i for i in range(1, int(np.ceil(np.log10(max_throughput))))]
        if algo_display == 'pgvector':
            yticks.append(10**3.5)
        elif algo_display == 'Milvus':
            # Add custom ticks: 50, 100, 200 for better granularity around 10**2
            yticks = [50, 10**2, 200]
        ax.set_yticks(yticks)
        # Force matplotlib to show all ticks (especially important for log scale)
        if algo_display == 'Milvus':
            ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.grid(True, which="both", ls="--", zorder=0)
    if not return_handles_labels:
        ax.legend(fontsize=24)
    
    if return_handles_labels:
        return handles, labels, metadata
    return None, None, []


def main():
    root_results = '/home/abylay/ann-benchmarks-HQ/results'
    dataset_sizes = ['small', 'medium', 'large']
    plots_dir = f"{root_results}/MoRe_UPD_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data for all dataset sizes and both att_idx values
    averages_dict = {}
    for dataset_size in dataset_sizes:
        averages_dict[dataset_size] = {}
        for att_idx in [0, 1]:
            csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_{att_idx}/all_results_hnsw.csv"
            try:
                df = pd.read_csv(csv_path)
                averages_dict[dataset_size][att_idx] = compute_averages(df)
            except FileNotFoundError:
                print(f"Error: CSV file not found at {csv_path}")
                averages_dict[dataset_size][att_idx] = None
    
    # Create plots: 2x3 layout (top row: movies, bottom row: reviews, columns: dataset sizes)
    # Two pictures: one for pgvector, one for Milvus
    algorithms = [('pgvector', 'pgvector'), ('milvus-hnsw', 'Milvus')]
    
    for algorithm, algo_display in algorithms:
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        all_metadata = []  # Collect metadata from all subplots
        
        # Top row: movies
        for col_idx, dataset_size in enumerate(dataset_sizes):
            if averages_dict[dataset_size][0] is not None or averages_dict[dataset_size][1] is not None:
                handles, labels, metadata = plot_throughput_vs_recall_by_selectivity_attidx(
                    axes[0, col_idx], 
                    averages_dict[dataset_size][0],
                    averages_dict[dataset_size][1],
                    dataset_size, 'movies', algorithm, 
                    return_handles_labels=True
                )
                all_metadata.extend(metadata)
        
        # Bottom row: reviews
        for col_idx, dataset_size in enumerate(dataset_sizes):
            if averages_dict[dataset_size][0] is not None or averages_dict[dataset_size][1] is not None:
                handles, labels, metadata = plot_throughput_vs_recall_by_selectivity_attidx(
                    axes[1, col_idx], 
                    averages_dict[dataset_size][0],
                    averages_dict[dataset_size][1],
                    dataset_size, 'reviews', algorithm, 
                    return_handles_labels=True
                )
                all_metadata.extend(metadata)
        
        # Create unified legend
        # Group by (att_idx, selectivity_index) -> collect selectivity values per query_type
        legend_data = {}  # (att_idx, sel_idx) -> {query_type: sel_value, handle: line}
        
        for handle, att_idx, sel_idx, sel_value, qt in all_metadata:
            key = (att_idx, sel_idx)
            if key not in legend_data:
                legend_data[key] = {'handle': handle, 'sel_values': {}}
            legend_data[key]['sel_values'][qt] = sel_value
        
        # Create legend entries
        legend_handles = []
        legend_labels = []
        
        # Sort by att_idx (0 first, then 1), then by selectivity index
        sorted_keys = sorted(legend_data.keys(), key=lambda x: (x[0], x[1]))
        
        for att_idx, sel_idx in sorted_keys:
            data = legend_data[(att_idx, sel_idx)]
            sel_dict = data['sel_values']
            # Format: att_idx=X σ={movies: val1, reviews: val2}
            sel_strs = []
            for qt in ['movies', 'reviews']:
                if qt in sel_dict:
                    sel_val = sel_dict[qt]
                    sel_str = f'{sel_val:.2f}' if sel_val >= 0.01 else f'{sel_val:.3f}'
                    sel_strs.append(f'{qt}: {sel_str}')
            sel_label = '{' + ', '.join(sel_strs) + '}'
            legend_handles.append(data['handle'])
            legend_labels.append(f'att_idx={att_idx}, $\\sigma_g$={sel_label}')
        
        # Place legend at the bottom center of the figure
        n_att_idx = len(set(att_idx for att_idx, _ in sorted_keys))
        n_entries_per_att = len(legend_handles) // n_att_idx if n_att_idx > 0 else len(legend_handles)
        fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=24)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(os.path.join(plots_dir, f'hnsw_throughput_vs_recall_by_selectivity_attidx_{algo_display.lower()}.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, f'hnsw_throughput_vs_recall_by_selectivity_attidx_{algo_display.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {plots_dir}/hnsw_throughput_vs_recall_by_selectivity_attidx_{algo_display.lower()}.pdf/png")

if __name__ == "__main__":
    main()

