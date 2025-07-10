import pandas as pd
import h5py
import os
import re
from pathlib import Path
import numpy as np

def calculate_recall(neighbors, true_neighbors):
    return len(set(neighbors).intersection(set(true_neighbors))) / len(set(true_neighbors))

def extract_idx_number(filename):
    match = re.search(r'M_(\d+)_ef', filename)
    return int(match.group(1)) if match else None

def load_true_neighbors(true_df_path):
    with h5py.File(true_df_path, 'r') as f:
        return f['neighbors'][:], f['distances'][:]
    

def process_hdf5_files(filters, k_vals, m_vals, expected_lines=250):
    # Initialize result storage
    result_data = []
    correct_gt = True
    
    # Walk through directory structure
    for filter in filters:
        for k in k_vals:
            
            # Load true neighbors for this filter and k combination
            true_df_path = f"data/dataset/dataset_imdbHQ_f{filter}_k{k}_upd.h5"
            true_df_path_2 = f"results/glove-100-angular/9/{k}/pgvector/angular_M_16_efConstruction_200_50.hdf5"
            true_neighbors, true_distances = load_true_neighbors(true_df_path)
            
            with h5py.File(true_df_path_2, 'r') as f:
                true_neighbors2 = f['neighbors'][:]
                true_distances2 = f['distances'][:]
            
            # Initialize temporary result rows by id
            temp_rows = {i: {'id': i, 'filter': filter, 'k': k} for i in range(expected_lines)}
            
            for m in m_vals:
                if filter == 0: filter = "None"
                file_path = f"results/glove-100-angular/{filter}/{k}/milvus-hnsw/angular_768_M_{m}_efConstruction_{m*4}_40.hdf5" # pgvec
                if filter == "None": filter = 0
                with h5py.File(file_path, 'r') as f:
                    neighbors = f['neighbors'][:]
                    distances = f['distances'][:]
                    times = f['times'][:]
                                    
                # Verify number of lines
                if len(neighbors) != expected_lines:
                    print(f"Warning: {file_path} has {len(neighbors)} lines, expected {expected_lines}")
                        
                # Process each line
                for i in range(len(neighbors)):
                    recall = calculate_recall(neighbors[i], true_neighbors[i])
                    true_recall_check = 1 - calculate_recall(true_neighbors[i], true_neighbors2[i]) # pgvec
                    temp_rows[i][f'{m}_recall'] = recall
                    if true_recall_check > 0.0001:
                        correct_gt = False
                        print(f"Not matching true_neighbor for row {i} with Recall: {1-true_recall_check}")
                        print(f"True neighbors: sklearn - {true_neighbors[i]} || pgvec - {true_neighbors2[i]}")
                        print(f"Farthest distances: sklearn - {max(true_distances[i])} || pgvec - {max(true_distances2[i])} || DIFF: {max(true_distances2[i]) - max(true_distances[i])}")
                    temp_rows[i][f'{m}_runtime'] = times[i]
            
            result_data.extend(temp_rows.values())
            # print(temp_rows)
    
    # Checking true result correctness
    if correct_gt: print("GROUND TRUTH RESULTS VERIFIED!")
    
    # Create DataFrame
    df = pd.DataFrame(result_data)
    print("Results shape:", df.shape)
    
    # Ensure all columns are present and properly formatted
    if not df.empty:
        # Group by id, filter, k and aggregate
        df = df.groupby(['id', 'filter', 'k']).first().reset_index()
        
        # Convert filter and k to appropriate types if needed
        df['filter'] = df['filter'].astype(str)
        df['k'] = df['k'].astype(str)
        
        # Aggregated data
        group_columns = ['filter', 'k']
        average_columns = ["4_recall", "4_runtime", "8_recall", "8_runtime", "16_recall", "16_runtime"]
        averages = df.groupby(group_columns)[average_columns].mean().reset_index()
        print(averages)

    print("Results shape:", df.shape)        
    return df

def main():
    # Process files and create DataFrame
    filters = [0]
    k_vals = [10]
    m_vals = [4] # [4, 8, 16]
    result_df = process_hdf5_files(filters, k_vals, m_vals)
    result_file_name = "all_results.csv"
    # Save or return the DataFrame
    if not result_df.empty:
        result_df.to_csv(f'results/{result_file_name}', index=False)
        print("Results saved to {result_file_name}")
    else:
        print("No data processed")

if __name__ == "__main__":
    main()
