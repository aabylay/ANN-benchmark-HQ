import pandas as pd
import h5py
import os
import re
from pathlib import Path
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ann_benchmarks.plotting.utils import compute_metrics
from ann_benchmarks.plotting.metrics import get_recall_values, epsilon_threshold, knn_threshold


def calculate_recall(neighbors, true_neighbors):
    return len(set(neighbors).intersection(set(true_neighbors))) / len(set(true_neighbors))

def extract_idx_number(filename):
    match = re.search(r'M_(\d+)_ef', filename)
    return int(match.group(1)) if match else None

def load_true_neighbors(true_df_path):
    with h5py.File(true_df_path, 'r') as f:
        return f['neighbors'][:], f['distances'][:]
    

def process_hdf5_files(filters, k_vals, m_vals, efs_vals, expected_lines=250):
    # Initialize result storage
    result_data = []
    
    # Walk through directory structure
    for filter in filters:
        for k in k_vals:
            
            # Load true neighbors for this filter and k combination
            true_df_path = f"data/dataset/dataset_imdbHQ_f{filter}_k{k}_upd.h5"
            true_neighbors, true_distances = load_true_neighbors(true_df_path)
            
            # Initialize temporary result rows by id
            
            for m in m_vals:
                for ef in efs_vals:
                    temp_rows = {i: {'id': i, 'filter': filter, 'k': k, 'ef': ef} for i in range(expected_lines)}
                    file_path = f"results/glove-100-angular/{filter}/{k}/pgvector/angular_M_{m}_efConstruction_{m*4}_{ef}.hdf5" # pgvec
                    print(f"Processing file: {file_path}")
                    with h5py.File(file_path, 'r') as f:
                        neighbors = f['neighbors'][:]
                        distances = f['distances'][:]
                        times = f['times'][:]
                                    
                    # Verify number of lines
                    if len(neighbors) != expected_lines:
                        print(f"Warning: {file_path} has {len(neighbors)} lines, expected {expected_lines}")
                        
                    # Calculate recall
                    runs = get_recall_values(true_distances, np.array(distances), k, knn_threshold, epsilon=1e-3)
                    print(f"RESULTS FOR: k={k}, filter={filter}, m={m}, ef={ef}")
                    print(f"Recall Mean: {runs[0]}, Recall StdDev: {runs[1]}")
                    print(f"Runtime Mean: {np.array(times).mean()}")
                    print("=============\n")

                    for i in range(len(neighbors)):                    
                        temp_rows[i][f'{m}_recall'] = runs[2][i]
                        temp_rows[i][f'{m}_runtime'] = times[i]
            
                    result_data.extend(temp_rows.values())
                    print(len(result_data), "rows processed so far...")
    
    print(result_data[-5:])
    # Create DataFrame
    df = pd.DataFrame(result_data)
    
    # Ensure all columns are present and properly formatted
    if not df.empty:
        # Group by id, filter, k and aggregate
        df = df.groupby(['id', 'filter', 'k', 'ef']).first().reset_index()
        
        # Convert filter and k to appropriate types if needed
        df['filter'] = df['filter'].astype(str)
        df['k'] = df['k'].astype(str)
        df['ef'] = df['ef'].astype(str)
        
        # Aggregated data
        group_columns = ['filter', 'k', 'ef']
        average_columns = ["4_recall", "4_runtime", "8_recall", "8_runtime", "16_recall", "16_runtime"]
        averages = df.groupby(group_columns)[average_columns].mean().reset_index()
        print(f"Summary for all results:\n {averages}")

    print("All results DataFrame shape:", df.shape)
    return df

def main():
    # Process files and create DataFrame
    filters = [0, 6, 6.5, 7, 7.5, 8, 8.3, 9, 9.5]
    k_vals = [1, 2, 5, 10, 20]
    m_vals = [4, 8, 16]
    efs_vals = [20, 40, 60, 80, 100, 200, 300, 400, 500]
    result_df = process_hdf5_files(filters, k_vals, m_vals, efs_vals)
    result_file_name = "all_results_pgvec.csv"
    # Save or return the DataFrame
    if not result_df.empty:
        result_df.to_csv(f'results/{result_file_name}', index=False)
        print("Results saved to {result_file_name}")
    else:
        print("No data processed")

if __name__ == "__main__":
    main()
