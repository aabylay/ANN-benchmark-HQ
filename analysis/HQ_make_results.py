import pandas as pd
import h5py
import os
import re
from pathlib import Path
import numpy as np
from ann_benchmarks.plotting.utils import compute_metrics


def calculate_recall(neighbors, true_neighbors):
    return len(set(neighbors).intersection(set(true_neighbors))) / len(set(true_neighbors))

def extract_idx_number(filename):
    match = re.search(r'M_(\d+)_ef', filename)
    return int(match.group(1)) if match else None

def load_true_neighbors(true_df_path, filter_val, k_val):
    with h5py.File(true_df_path, 'r') as f:
        return f['neighbors'][:], f['distances'][:]
    

def process_hdf5_files(base_path, expected_lines=250):
    # Initialize result storage
    result_data = []
    
    # Walk through directory structure
    for root, _, files in os.walk(base_path):
        print(f"\nFiles found: {files}")
        path_parts = Path(root).relative_to(base_path).parts
        print(path_parts)
        if len(path_parts) == 4:
            filter_val = path_parts[-3]
            k_val = path_parts[-2]
            print(f"Filter:", filter_val, "--- k-value:", k_val)
            # Load true neighbors for this filter and k combination
            true_df_path = f"data/dataset/dataset_imdbHQ_f{filter_val}_k{k_val}_upd.h5"
            if not os.path.exists(true_df_path): continue                
            else: print(f"Found true results for dataset_imdbHQ_f{filter_val}_k{k_val}_upd")

            true_neighbors, true_distances = load_true_neighbors(true_df_path, filter_val, k_val)
            
            for filename in files:
                print("   Filename:", filename)
                if filename.endswith('.hdf5') and 'M_' in filename:
                    idx_number = extract_idx_number(filename)
                    if idx_number is None:
                        print(f"... no idx edge degree found for {filename} ...")
                        continue
                    else: print(f"VALID idx found {filename}!")
                    file_path = os.path.join(root, filename)
                    with h5py.File(file_path, 'r') as f:
                        neighbors = f['neighbors'][:]
                        distances = f['distances'][:]
                        times = f['times'][:]
                        
                        # Verify number of lines
                        if len(neighbors) != expected_lines:
                            print(f"Warning: {file_path} has {len(neighbors)} lines, expected {expected_lines}")
                            continue
                            
                        # Process each line
                        for line_idx in range(len(neighbors)):
                            #print(type(filter_val), type(k_val), type(idx_number))
                            if filter_val == "0" and k_val == "5" and idx_number == 64:
                                print("Results:", distances[line_idx])
                                print("TrueRes:", true_distances[line_idx])
                                print("Results:", neighbors[line_idx])
                                print("TrueRes:", true_neighbors[line_idx], end="\n\n")
                            
                            recall = calculate_recall(neighbors[line_idx], true_neighbors[line_idx])
                            runs = compute_metrics(true_distances, np.array(distances), "k-nn", "qps", True)
                            result_data.append({
                                'id': line_idx,
                                'filter': filter_val,
                                'k': k_val,
                                f'{idx_number}_recall': recall,
                                f'{idx_number}_runtime': times[line_idx]
                            })
                else: print("   Incorrect filename:", filename)
    
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

    print("Results shape:", df.shape)        
    return df

def main():
    base_path = f"results/"
    
    # Process files and create DataFrame
    result_df = process_hdf5_files(base_path)
    result_file_name = "all_results.csv"
    # Save or return the DataFrame
    if not result_df.empty:
        result_df.to_csv(f'results/{result_file_name}', index=False)
        print("Results saved to {result_file_name}")
    else:
        print("No data processed")

if __name__ == "__main__":
    main()
