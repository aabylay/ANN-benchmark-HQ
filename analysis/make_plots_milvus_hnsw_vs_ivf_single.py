import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Set matplotlib to use DejaVu Serif (LaTeX-like font) and configure mathtext
mpl.rcParams['lines.linewidth'] = 2
plt.rc('font', family='serif', serif='DejaVu Serif', size=18)
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

def main():
    root_results = '/home/abylay/ann-benchmarks-HQ/results'
    dataset_size = 'large'
    query_type = 'movies'
    plots_dir = f"{root_results}/MoRe_UPD_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load HNSW data
    hnsw_csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_0/all_results_hnsw.csv"
    try:
        df_hnsw = pd.read_csv(hnsw_csv_path)
        hnsw_averages = compute_averages_hnsw(df_hnsw)
    except FileNotFoundError:
        print(f"Error: HNSW CSV file not found at {hnsw_csv_path}")
        return
    
    # Load IVF data
    ivf_csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_0/all_results_ivf.csv"
    try:
        df_ivf = pd.read_csv(ivf_csv_path)
        ivf_averages = compute_averages_ivf(df_ivf)
    except FileNotFoundError:
        print(f"Error: IVF CSV file not found at {ivf_csv_path}")
        return
    
    # Filter for Milvus and movies
    hnsw_sub = hnsw_averages[(hnsw_averages['query_type'] == query_type) & 
                             (hnsw_averages['k'] == 10) & 
                             (hnsw_averages['m'] == 10) &
                             (hnsw_averages['algorithm'] == 'milvus-hnsw')]
    
    ivf_sub = ivf_averages[(ivf_averages['query_type'] == query_type) & 
                           (ivf_averages['k'] == 10) &
                           (ivf_averages['algorithm'] == 'milvus-ivfflat')]
    
    # Get selectivities
    hnsw_sel = sorted(hnsw_sub['filter_selectivity'].unique()) if len(hnsw_sub) > 0 else []
    ivf_sel = sorted(ivf_sub['filter_selectivity'].unique()) if len(ivf_sub) > 0 else []
    all_sel = sorted(set(hnsw_sel + ivf_sel))
    unique_sel = [all_sel[i] for i in [1, 3, 5, -1] if i < len(all_sel)]
    
    markers_sel = ['.', '+', 'x', 's']
    blue_colors = ["#8888FF", "#5555FF", "#3333FF", "#0000FF"]
    red_colors = ["#FF8888", "#FF5555", "#FF3333", "#FF0000"]
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot HNSW curves (blue)
    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel = hnsw_sub[hnsw_sub['filter_selectivity'] == sel]
        if len(sub_sel) > 0:
            grouped = sub_sel.groupby('ef_search').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label = f'HNSW selectivity≈{sel_approx}'
            ax.plot(grouped['recall'], grouped['throughput'], marker=markers_sel[idx], 
                   color=blue_colors[idx], label=label, linestyle='-', linewidth=2)
    
    # Plot IVF curves (red)
    for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
        sub_sel = ivf_sub[ivf_sub['filter_selectivity'] == sel]
        if len(sub_sel) > 0:
            grouped = sub_sel.groupby('probes').agg({'recall': 'mean', 'throughput': 'mean'}).reset_index()
            grouped = grouped.sort_values(by='recall')
            sel_approx = f'{sel:.2f}' if sel >= 0.01 else f'{sel:.3f}'
            label = f'IVF selectivity≈{sel_approx}'
            ax.plot(grouped['recall'], grouped['throughput'], marker=markers_sel[idx], 
                   color=red_colors[idx], label=label, linestyle='--', linewidth=2)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([10**0, 10**4])
    ax.set_xlabel('Recall', fontsize=18)
    ax.set_ylabel('QPS', fontsize=18)
    ax.set_title(f'Milvus: HNSW vs IVF, Movies, @k=10, (large)', fontsize=20)
    ax.set_yscale('log')
    
    # Set y-ticks based on data range
    all_throughput = []
    if len(hnsw_sub) > 0:
        all_throughput.extend(hnsw_sub['throughput'].values)
    if len(ivf_sub) > 0:
        all_throughput.extend(ivf_sub['throughput'].values)
    if len(all_throughput) > 0:
        max_throughput = max(all_throughput)
        ax.set_yticks([10**i for i in range(1, int(np.ceil(np.log10(max_throughput))) + 1)])
    
    ax.grid(True, which="both", ls="--")
    ax.legend(fontsize=14, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'milvus_hnsw_vs_ivf_movies_large.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plots_dir}/milvus_hnsw_vs_ivf_movies_large.png")

if __name__ == "__main__":
    main()

