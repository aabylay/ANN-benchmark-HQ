import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Set matplotlib to use DejaVu Serif (LaTeX-like font) and configure mathtext
mpl.rcParams['lines.linewidth'] = 2
plt.rc('font', family='serif', serif='DejaVu Serif', size=32)
plt.rc('mathtext', default='regular')

def plot_index_build_times(plots_dir):
    """Plot: Bar plot for index build times - split into two subplots"""
    # Color scheme matching make_plots_poster.py
    colors_sel = [
        ["#FF8888", "#FF5555", "#3333FF", "#FF0000"],  # milvus-hnsw (red shades)
        ["#8888FF", "#5555FF", "#3333FF", "#0000FF"],  # pgvector (blue shades)
        ["#77DD77", "#55DD55", "#33DD33", "#11DD11"]   # hnsw(faiss) (green shades)
    ]
    
    build_times = {
        '551155': {
            'Milvus': [74.204204082489, 83.4855630397796, 90.5670397281646],
            'pgvector': [309.425403118133, 498.510179519653, 890.412653446197],
            'FAISS': [615.876474380493, 1851.26402044296, 3184.96159100532]
        },
        '2598267': {
            'Milvus': [326.468571901321, 382.14872932434, 431.08714556694],
            'pgvector': [2306.80065870285, 5670.61093235015, 11823.6678855419],
            'FAISS': [2898.6007797718, 8676.34518194198, 15794.1386625766]
        }
    }
    M_values = [5, 10, 15]
    algorithms = ['Milvus', 'pgvector', 'FAISS']
    bar_width = 0.24
    datasets = ['551155', '2598267']
    dataset_names = ['Movies', 'Reviews']

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    legend_lines = []
    legend_labels = []

    # Plot 1: Movies dataset (551155)
    x1 = np.arange(len(M_values))
    for algo_idx, algo in enumerate(algorithms):
        bars = ax1.bar(x1 + algo_idx * bar_width, build_times['551155'][algo], 
                width=bar_width, label=algo, 
                color=colors_sel[algo_idx][0])
        # Add to legend (only once, from first subplot)
        legend_lines.append(bars[0])
        legend_labels.append(algo)
    
    ax1.set_ylim(10**1, 10**4.5)
    ax1.set_xlabel('M - max degree of the graph')
    ax1.set_ylabel('Index Build Time (s)')
    ax1.set_title(f'{dataset_names[0]}')
    ax1.set_xticks(x1 + (len(algorithms) - 1) / 2 * bar_width)
    ax1.set_xticklabels(M_values)
    ax1.set_yticks([10**i for i in range(1, 5)])
    ax1.set_yscale('log')
    ax1.tick_params(axis='y')
    ax1.grid(True, which="both", ls="--")

    # Plot 2: Reviews dataset (2598267)
    x2 = np.arange(len(M_values))
    for algo_idx, algo in enumerate(algorithms):
        bars = ax2.bar(x2 + algo_idx * bar_width, build_times['2598267'][algo], 
                width=bar_width, label=algo, 
                color=colors_sel[algo_idx][0])
    ax2.set_ylim(10**1, 10**4.5)
    ax2.set_xlabel('M - max degree of the graph')
    ax2.set_ylabel('Index Build Time (s)')
    ax2.set_title(f'{dataset_names[1]}')
    ax2.set_xticks(x2 + (len(algorithms) - 1) / 2 * bar_width)
    ax2.set_xticklabels(M_values)
    ax2.set_yticks([10**i for i in range(1, 5)])
    ax2.set_yscale('log')
    ax2.tick_params(axis='y')
    ax2.grid(True, which="both", ls="--")
    
    # Add common legend at the bottom
    fig.legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=3)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save as PNG
    png_path = os.path.join(plots_dir, 'index_build_times.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"PNG plot saved: {png_path}")
    
    # Save as PDF
    pdf_path = os.path.join(plots_dir, 'index_build_times.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF plot saved: {pdf_path}")
    
    plt.close()

def main():
    root_results = '/home/abylay/ann-benchmarks-HQ/results'
    plots_dir = f"{root_results}/MoRe_UPD_plots"
    plot_index_build_times(plots_dir)

if __name__ == "__main__":
    main()

