import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Exact cardinalities for movies and reviews datasets
data_sizes = {
    "small": {
        "movies": "small",
        "reviews": "small"
    },
    "medium": {
        "movies": "medium",
        "reviews": "medium"
    },
    "large": {
        "movies": "large",
        "reviews": "large"
    }
}

# Algorithm name mappings
hnsw_algo_names = {
    'milvus-hnsw': 'Milvus',
    'pgvector': 'pgvector',
    'hnsw(faiss)': 'FAISS'
}

ivf_algo_names = {
    'milvus-ivfflat': 'Milvus',
    'pgvector_ivf': 'pgvector',
    'faiss-ivf': 'FAISS'
}

# System order for subplots
system_order = ['Milvus', 'pgvector', 'FAISS']

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

def plot_hnsw_ivf_comparison(ax, hnsw_averages, ivf_averages, dataset_size, query_type='movies', system_name='Milvus', return_handles_labels=False):
    """Plot HNSW vs IVF QPS vs Recall curves by selectivity for a specific system"""
    markers_sel = ['.', '+', 'x', 's']
    
    # System-based colors: pgvector=blue, milvus=red, faiss=green
    system_colors = {
        'pgvector': '#0000FF',  # Blue
        'Milvus': '#FF0000',    # Red
        'FAISS': '#00FF00'      # Green
    }
    
    # Darker/greyer versions for IVF (dashed lines)
    system_colors_ivf = {
        'pgvector': '#6666AA',  # Darker blue-grey
        'Milvus': '#AA6666',    # Darker red-grey
        'FAISS': '#66AA66'      # Darker green-grey
    }
    
    system_color = system_colors.get(system_name, '#000000')
    system_color_ivf = system_colors_ivf.get(system_name, '#666666')
    
    # Linewidth multipliers for selectivities: [0.75, 1, 1.4, 2]
    base_linewidth = 2
    linewidth_multipliers = [0.8, 0.9, 1.4, 2]
    
    # Map system name to algorithm names
    hnsw_algo_map = {'Milvus': 'milvus-hnsw', 'pgvector': 'pgvector', 'FAISS': 'hnsw(faiss)'}
    ivf_algo_map = {'Milvus': 'milvus-ivfflat', 'pgvector': 'pgvector_ivf', 'FAISS': 'faiss-ivf'}
    
    hnsw_algo = hnsw_algo_map.get(system_name)
    ivf_algo = ivf_algo_map.get(system_name)
    
    # Filter HNSW data
    hnsw_sub = hnsw_averages[(hnsw_averages['query_type'] == query_type) & 
                             (hnsw_averages['k'] == 10) & 
                             (hnsw_averages['m'] == 10) &
                             (hnsw_averages['algorithm'] == hnsw_algo)]
    
    # Filter IVF data
    ivf_sub = ivf_averages[(ivf_averages['query_type'] == query_type) & 
                           (ivf_averages['k'] == 10) &
                           (ivf_averages['algorithm'] == ivf_algo)]
    
    # Get selectivities (use HNSW selectivities as reference, or combine both)
    hnsw_sel = sorted(hnsw_sub['filter_selectivity'].unique()) if len(hnsw_sub) > 0 else []
    ivf_sel = sorted(ivf_sub['filter_selectivity'].unique()) if len(ivf_sub) > 0 else []
    all_sel = sorted(set(hnsw_sel + ivf_sel))
    unique_sel = [all_sel[i] for i in [1, 3, 5, -1] if i < len(all_sel)]
    
    handles = []
    labels = []
    metadata = []  # Store (handle, index_type, sel_idx, sel_value, query_type)
    
    # Plot HNSW curves (solid lines, system color)
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
    
    # Plot IVF curves (dashed lines, darker/greyer system color)
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
    
    # Add vertical line at recall 1.0 in the background
    ylim = ax.get_ylim()
    ax.axvline(x=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
    ax.set_ylim(ylim)  # Restore ylim after axvline
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    ax.set_title(f'{query_type.capitalize()}, @k=10, ({data_sizes[dataset_size][query_type]})')
    ax.set_yscale('log')
    
    # Set y-ticks based on data range
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
    
    # Load data for all dataset sizes
    hnsw_averages_dict = {}
    ivf_averages_dict = {}
    
    for dataset_size in dataset_sizes:
        # Load HNSW data
        hnsw_csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_0/all_results_hnsw.csv"
        try:
            df_hnsw = pd.read_csv(hnsw_csv_path)
            hnsw_averages_dict[dataset_size] = compute_averages_hnsw(df_hnsw)
        except FileNotFoundError:
            print(f"Error: HNSW CSV file not found at {hnsw_csv_path}")
            hnsw_averages_dict[dataset_size] = None
        
        # Load IVF data
        ivf_csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_0/all_results_ivf.csv"
        try:
            df_ivf = pd.read_csv(ivf_csv_path)
            ivf_averages_dict[dataset_size] = compute_averages_ivf(df_ivf)
        except FileNotFoundError:
            print(f"Error: IVF CSV file not found at {ivf_csv_path}")
            ivf_averages_dict[dataset_size] = None
    
    # Create separate figure for each system
    for system_name in system_order:
        # Create 2x3 figure (2 rows: movies, reviews; 3 columns: dataset sizes)
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        all_metadata = []  # Collect metadata from all subplots
        
        for row_idx, query_type in enumerate(['movies', 'reviews']):
            for col_idx, dataset_size in enumerate(dataset_sizes):
                if hnsw_averages_dict[dataset_size] is None or ivf_averages_dict[dataset_size] is None:
                    print(f"Skipping {dataset_size} dataset - missing data")
                    continue
                
                handles, labels, metadata = plot_hnsw_ivf_comparison(
                    axes[row_idx, col_idx], 
                    hnsw_averages_dict[dataset_size], 
                    ivf_averages_dict[dataset_size], 
                    dataset_size, 
                    query_type=query_type,
                    system_name=system_name,
                    return_handles_labels=True
                )
                all_metadata.extend(metadata)
        
        # Create unified legend
        # Group by (index_type, selectivity_index) -> collect selectivity values per query_type
        legend_data = {}  # (index_type, sel_idx) -> {query_type: sel_value, handle: line}
        
        for handle, index_type, sel_idx, sel_value, query_type in all_metadata:
            key = (index_type, sel_idx)
            if key not in legend_data:
                legend_data[key] = {'handle': handle, 'sel_values': {}}
            legend_data[key]['sel_values'][query_type] = sel_value
        
        # Create legend entries
        legend_handles = []
        legend_labels = []
        
        # Sort by index type (HNSW first, then IVF), then by selectivity index
        sorted_keys = sorted(legend_data.keys(), key=lambda x: (x[0], x[1]))
        
        for index_type, sel_idx in sorted_keys:
            data = legend_data[(index_type, sel_idx)]
            sel_dict = data['sel_values']
            # Format: IndexType σ={movies: val1, reviews: val2}
            sel_strs = []
            for qt in ['movies', 'reviews']:
                if qt in sel_dict:
                    sel_val = sel_dict[qt]
                    sel_str = f'{sel_val:.2f}' if sel_val >= 0.01 else f'{sel_val:.3f}'
                    sel_strs.append(f'{qt}: {sel_str}')
            sel_label = '{' + ', '.join(sel_strs) + '}'
            legend_handles.append(data['handle'])
            legend_labels.append(f'{index_type} $\\sigma_g$={sel_label}')
        
        # Place legend at the bottom center of the figure
        # Calculate optimal number of columns
        n_index_types = len(set(idx_type for idx_type, _ in sorted_keys))
        n_entries_per_type = len(legend_handles) // n_index_types if n_index_types > 0 else len(legend_handles)
        fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=24)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(os.path.join(plots_dir, f'hnsw_vs_ivf_comparison_{system_name.lower()}.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, f'hnsw_vs_ivf_comparison_{system_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {plots_dir}/hnsw_vs_ivf_comparison_{system_name.lower()}.pdf/png")

if __name__ == "__main__":
    main()

