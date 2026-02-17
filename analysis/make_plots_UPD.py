import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, argparse

def compute_averages(df):
    df['query_type'] = df['query_id'].apply(lambda x: 'movies' if x.startswith('qm') else 'reviews')
    averages = df.groupby(['query_type', 'filter_id', 'filter_selectivity', 'k', 'ef_search', 'algorithm', 'm'])[['recall', 'runtime']].mean().reset_index()
    averages['throughput'] = 1 / averages['runtime']
    return averages

def plot_throughput_vs_recall_by_ef(averages, plots_dir, att_idx):
    averages = averages[averages['k'].isin([10])]
    for qt in averages['query_type'].unique():
        plt.figure(figsize=(10, 6))
        sub_qt = averages[averages['query_type'] == qt]
        for j, algo in enumerate(sub_qt['algorithm'].unique()):
            for i, m in enumerate(sub_qt['m'].unique()):
                sub = sub_qt[(sub_qt['algorithm'] == algo) & (sub_qt['m'] == m)]
                grouped = sub.groupby('ef_search').agg({
                    'recall': 'mean',
                    'throughput': 'mean'
                }).reset_index()
                grouped = grouped.sort_values(by='ef_search')
                label = f'{algo} m={m}'
                if algo == 'pgvector': algo_idx = 1
                elif algo == 'milvus-hnsw': algo_idx = 0
                elif algo == 'hnsw(faiss)': algo_idx = 2
                plt.plot(grouped['recall'], grouped['throughput'], marker=markers[i], color=colors[algo_idx][i], label=label)
        plt.xlabel('Recall')
        plt.ylabel('Throughput (queries per second)')
        plt.title(f'Throughput vs Recall by ef_search (att_idx={att_idx}, @k=10) - {qt}')
        plt.yscale('log')
        plt.yticks([10**i for i in range(1, int(np.ceil(np.log10(max(sub_qt['throughput'])))) + 1)])
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(plots_dir, f'throughput_vs_recall_by_ef_{qt}_attidx_{att_idx}.png'))
        plt.close()

def plot_recall_vs_ef_by_selectivity(averages, plots_dir, att_idx):
    averages = averages[averages['k'].isin([10])]
    averages = averages[averages['m'].isin([10])]
    for qt in averages['query_type'].unique():
        plt.figure(figsize=(10, 6))
        sub_qt = averages[averages['query_type'] == qt]
        unique_sel = sorted(sub_qt['filter_selectivity'].unique())
        for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
            sub_sel = sub_qt[sub_qt['filter_selectivity'] == sel]
            for j, algo in enumerate(sub_sel['algorithm'].unique()):
                if algo == 'milvus-hnsw': algo_idx = 0
                elif algo == 'pgvector': algo_idx = 1
                else: algo_idx = 2
                sub = sub_sel[sub_sel['algorithm'] == algo]
                grouped = sub.groupby('ef_search').agg({
                    'recall': 'mean'
                }).reset_index()
                grouped = grouped.sort_values(by='ef_search')
                label = f'{algo} sel={sel:.2e}'
                plt.plot(grouped['ef_search'], grouped['recall'], marker=markers_sel[idx], color=colors_sel[algo_idx][idx], label=label)
        plt.xlabel('ef_search')
        plt.ylabel('Recall')
        plt.title(f'Recall vs ef_search by selectivity (att_idx={att_idx}, @m=10, @k=10) - {qt}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'recall_vs_ef_by_selectivity_{qt}_attidx_{att_idx}.png'))
        plt.close()

def plot_throughput_vs_recall_by_selectivity_fixed_k_m(averages, plots_dir, att_idx):
    
    print(averages["m"].unique())
    print(averages["k"].unique())
    averages = averages[averages['m'].isin([5, 15])]
    for qt in averages['query_type'].unique():
        sub_qt = averages[averages['query_type'] == qt]
        for k_val in [1, 10, 100]:
            for m_val in [5, 15]:
                plt.figure(figsize=(10, 6))
                sub = sub_qt[(sub_qt['k'] == k_val) & (sub_qt['m'] == m_val)]
                unique_sel = sorted(sub['filter_selectivity'].unique())
                # choose [0, 2, 4, -1] selectivities for plotting
                unique_sel = [unique_sel[i] for i in [0, 2, 4, -1] if i < len(unique_sel)]
                
                for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
                    sub_sel = sub[sub['filter_selectivity'] == sel]
                    for j, algo in enumerate(sub_sel['algorithm'].unique()):
                        if algo == 'milvus-hnsw': color = '#FF0000'
                        elif algo == 'pgvector': color = '#0000FF'
                        else: color = '#00FF00'
                        sub_algo = sub_sel[sub_sel['algorithm'] == algo]
                        grouped = sub_algo.groupby('ef_search').agg({
                            'recall': 'mean',
                            'throughput': 'mean'
                        }).reset_index()
                        grouped = grouped.sort_values(by='recall')
                        label = f'{algo} sel={sel:.2e}'
                        if algo == 'pgvector': algo_idx = 1
                        elif algo == 'milvus-hnsw': algo_idx = 0
                        elif algo == 'hnsw(faiss)': algo_idx = 2
                        plt.plot(grouped['recall'], grouped['throughput'], marker=markers_sel[idx], color=colors_sel[algo_idx][idx], label=label)
                plt.xlabel('Recall')
                plt.ylabel('Throughput (queries per second)')
                plt.title(f'Throughput vs Recall by selectivity, @k={k_val}, @m={m_val}, @att_idx={att_idx} - {qt}')
                plt.yscale('log')
                plt.yticks([10**i for i in range(1, int(np.ceil(np.log10(max(sub['throughput'])))) + 1)])
                plt.legend()
                plt.grid(True, which="both", ls="--")
                plt.savefig(os.path.join(plots_dir, f'throughput_vs_recall_by_selectivity_k_{k_val}_m_{m_val}_{qt}_attidx_{att_idx}.png'))
                plt.close()

def plot_throughput_vs_recall_by_selectivity_system(averages_dict, plots_dir, dataset_size):
    for algo in averages_dict[0]['algorithm'].unique():
        for qt in averages_dict[0]['query_type'].unique():
            for k_val in [10]:
                for m_val in [5, 15]:
                    plt.figure(figsize=(10, 6))
                    for att_idx in [0]:
                        averages = averages_dict[att_idx]
                        sub_qt = averages[(averages['query_type'] == qt) & (averages['algorithm'] == algo) & (averages['k'] == k_val) & (averages['m'] == m_val)]
                        unique_sel = sorted(sub_qt['filter_selectivity'].unique())
                        for idx, sel in enumerate(unique_sel[:len(markers_sel)]):
                            sub_sel = sub_qt[sub_qt['filter_selectivity'] == sel]
                            grouped = sub_sel.groupby('ef_search').agg({
                                'recall': 'mean',
                                'throughput': 'mean'
                            }).reset_index()
                            grouped = grouped.sort_values(by='recall')
                            label = f'{"with" if att_idx == 1 else "without"} attr index sel={sel:.2e}'
                            plt.plot(grouped['recall'], grouped['throughput'], marker=markers_sel[idx], color=colors_sel[att_idx][idx], label=label)
                    plt.xlabel('Recall')
                    plt.ylabel('Throughput (queries per second)')
                    plt.title(f'Throughput vs Recall by selectivity, @k={k_val}, @m={m_val} - {algo} - {qt} - {dataset_size}')
                    plt.yscale('log')
                    max_throughput = max(averages_dict[0][(averages_dict[0]['query_type'] == qt) & (averages_dict[0]['algorithm'] == algo) & (averages_dict[0]['k'] == k_val) & (averages_dict[0]['m'] == m_val)]['throughput'])
                    plt.yticks([10**i for i in range(1, int(np.ceil(np.log10(max_throughput))) + 1)])
                    plt.legend()
                    plt.grid(True, which="both", ls="--")
                    plt.savefig(os.path.join(plots_dir, f'throughput_vs_recall_by_selectivity_{algo}_k_{k_val}_m_{m_val}_{qt}_{dataset_size}.png'))
                    plt.close()
                    
def main(dataset_size):
    root_results = '/home/abylay/ann-benchmarks-HQ/results'
    plots_dir = f"{root_results}/MoRe_UPD_{dataset_size}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    averages_dict = {}
    
    for att_idx in [0]:
        csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_{att_idx}/all_results_hnsw.csv"
        
        df = pd.read_csv(csv_path)
        averages = compute_averages(df)
        averages_dict[att_idx] = averages
        
        plot_throughput_vs_recall_by_ef(averages, plots_dir, att_idx)
        plot_recall_vs_ef_by_selectivity(averages, plots_dir, att_idx)
        plot_throughput_vs_recall_by_selectivity_fixed_k_m(averages, plots_dir, att_idx)
        
        print(f"Plots for att_idx={att_idx} have been saved in:", plots_dir)
    
    # Create system-specific plots
    plot_throughput_vs_recall_by_selectivity_system(averages_dict, plots_dir, dataset_size)
    print(f"System-specific plots have been saved in:", plots_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset results")
    parser.add_argument("--dataset_size", choices=["small", "medium", "large"], default="small", help="Size of the dataset")
    args = parser.parse_args()
    markers = ['.', 'o', 's']
    colors = [["#FF6666", "#FF3333", "#FF0000"], ["#6666FF", "#3333FF", "#0000FF"], ["#66FF66", "#33FF33", "#00FF00"]]
    
    markers_sel = ['.', '+', 'X', 's']
    colors_sel = [
        ["#FF8888", "#FF5555", "#FF3333", "#FF0000"], 
        ["#8888FF", "#5555FF", "#3333FF", "#0000FF"],
        ["#88FF88", "#55FF55", "#33FF33", "#00FF00"],
        
    ]
    
    main(dataset_size=args.dataset_size)