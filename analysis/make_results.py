"""
INITIAL PROMPT:

Make plots for the results of ANN benchmark with filters.
Experiment results are stored in HDF5 files, and the code should read these files, extract relevant data, and generate plots to visualize the performance of different algorithms under various conditions.
Absolute path to the file: /home/abylay/ann-benchmarks-HQ/results/More_{dataset_size}/fid{filter_id}/{k}/{algo}/{"movies"/"reviews}_angular_{D}_M_{M}_efConstruction_{ef_construction}_{ef_search}.hdf5
Join everything into single csv file with following columns: 
- query_id: either qm**** or qr****depending on {"movies"/"reviews"}
- filter_id and filter selectivity (2 columns)
- k
- ef_search
- algorithm (milvus-hnsw / pgvector) and value of m (I believe only graph algorithms will be tested)
- recall, runtime (2 columns)
Each combination of query_id, filter_id, k, ef_search, algorithm should be a separate row.
Filter selectivities in file /home/abylay/ann-benchmarks-HQ/data/datasets/More_{dataset_size}/filters/{"movies"/"reviews}_filters_0.hdf5
Previous version of the code worked on different file structure, you can use it as a reference (below).
Unlike in the previous version, here we don't provide values of filters, k_vals, m_vals, efs_vals, expected_lines, etc. Just read all the files and extract corresponding values from filepath.

-- CODE --
"""

import pandas as pd
import h5py
import os, re, argparse
from pathlib import Path
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ann_benchmarks.plotting.utils import compute_metrics
from ann_benchmarks.plotting.metrics import get_recall_values, epsilon_threshold, knn_threshold


def parse_filename(file: Path) -> dict | None:
    filename_list = file.name.split('/')
    # Extract parameters from the filename and directory structure.
    filename_pattern = r"(movies|reviews)_angular_(\d+)_M_(\d+)_efConstruction_(\d+)_(\d+)\.hdf5"
    filename_pattern_pg = r"(movies|reviews)_angular_M_(\d+)_efConstruction_(\d+)_(\d+)\.hdf5"
    filename_pattern_mivf = r"(movies|reviews)_angular_(\d+)_nlist_(\d+)_(\d+)\.hdf5"
    filename_pattern_ivf = r"(movies|reviews)_angular_clusters_(\d+)_(\d+)\.hdf5"
    
    idx_type = None
    match = re.match(filename_pattern, file.name)
        
    if not match:
        match = re.match(filename_pattern_pg, file.name)
        if not match:
            match = re.match(filename_pattern_mivf, file.name)
            if not match:
                match = re.match(filename_pattern_ivf, file.name)
                if not match:
                    print(f"Filename {file.name} does not match expected pattern.")
                    print(f"Expected pattern milvus-hnsw: {filename_pattern}")
                    print(f"Expected pattern pg-vector / faiss: {filename_pattern_pg}")
                    print(f"Expected pattern ivf1: {filename_pattern_mivf}")
                    print(f"Expected pattern ivf2: {filename_pattern_ivf}")
                    sys.exit(1)
                    return None
                else:
                    # Extract query type, dimension, clusters, probes from the filename (faiss-ivf)
                    query_type = match.group(1)
                    dimension = ""
                    clusters = int(match.group(2))
                    probes = int(match.group(3))
                    idx_type = "ivf"
            else:
                # Extract query type, dimension from the filename (ivf)
                query_type = match.group(1)
                dimension = int(match.group(2))
                clusters = int(match.group(3))
                probes = int(match.group(4))
                idx_type = "ivf"
        else:
            # Extract query type, dimension, m, ef_construction, ef_search from the filename (pgvector)
            query_type = match.group(1)
            dimension = ""
            m = int(match.group(2))
            ef_construction = int(match.group(3))
            ef_search = int(match.group(4))
            idx_type = "hnsw"
    else:    
        # Extract query type, dimension, m, ef_construction, ef_search from the filename (milvus-hnsw)
        query_type = match.group(1)
        dimension = int(match.group(2))
        m = int(match.group(3))
        ef_construction = int(match.group(4))
        ef_search = int(match.group(5))
        idx_type = "hnsw"
    
    # Extract algorithm, k, filter_id, dataset_size from the directory structure
    algo = file.parent.name
    k = int(file.parent.parent.name)
    fid_match = re.match(r'fid(\d+)', file.parent.parent.parent.name)
    filter_id = int(fid_match.group(1))    
    
    if idx_type == "hnsw":
        return idx_type, {
            'query_type': query_type,
            'dimension': dimension,
            'm': m,
            'ef_construction': ef_construction,
            'ef_search': ef_search,
            'algo': algo,
            'k': k,
            'filter_id': filter_id,
        }
        
    elif idx_type == "ivf":
        return idx_type, {
            'query_type': query_type,
            'dimension': dimension,
            'clusters': clusters,
            'probes': probes,
            'algo': algo,
            'k': k,
            'filter_id': filter_id,
        }
        

def load_selectivities(root_data: Path, dataset_size: str, query_type: str) -> np.ndarray:
    filepath = root_data / f"MoRe_{dataset_size}" / "filters" / f"{query_type}_filters_0.hdf5"
    with h5py.File(filepath, 'r') as f:
        return f['selectivities'][:]


def load_true_distances(root_data: Path, dataset_size: str, data_table: str, filter_id: int, k: int) -> np.ndarray:
    true_path = root_data / f"MoRe_{dataset_size}" / f"queries" / f"queries_flex_{data_table}_sim_0_{filter_id}.hdf5"
    print(f"Loading true distances from: {str(true_path)}, --- {type(true_path)}")
    true_dists = []
    with h5py.File(true_path, 'r') as f:
        # Return only the first k distances for each query
        print(f"True distances shape: {f['distances'].shape}, {f['distances'][:][:k].shape}")
        return f['distances'][:, :k]
        

def process_single_file(file: Path, params: dict, selectivities: np.ndarray, true_distances: np.ndarray, idx_type: str) -> list[dict]:
    # Read distances and times from the file and verify the number of queries  
    
    print(f"Reading file: {file}")
    with h5py.File(file, 'r') as f:
        distances = f['distances'][:]
        times = f['times'][:]
    
    print("Verifying number of queries...")    
    num_queries = len(distances)
    if len(true_distances) != num_queries:
        raise ValueError(f"Mismatch in number of queries for {file}: expected {len(true_distances)}, got {num_queries}")

    # Prepare and verify filters and selectivities and calculate recall values
    filter_id = params['filter_id']
    if filter_id >= len(selectivities): 
        raise IndexError(f"Filter ID {filter_id} out of range for selectivities")
    filter_selectivity = selectivities[filter_id]
    
    file_name = str(file).split('/')
    """
    if "hnsw(faiss)" in file_name:
        print(true_distances)
        print(distances)
        raise Exception("Debugging")
    """
    
    try:
        print("Computing recall values...")
        runs = get_recall_values(true_distances, distances, params['k'], knn_threshold, epsilon=1e-6)
    except Exception as e:
        print(f"Error computing recall for file {file}: {e}")
        print("Distances", true_distances.shape, distances.shape)
        sys.exit(1)
    recalls = runs[2]
    
    print(f"Processed: {file}")
    print(f"Recall Mean: {runs[0]}, Recall StdDev: {runs[1]}")
    print(f"Runtime Mean: {np.mean(times)}")
    print("=============\n")
    
    # Prepare the result rows
    rows = []
    if idx_type == "hnsw":
        for i in range(num_queries):
            query_id = f"q{'m' if params['query_type'] == 'movies' else 'r'}{i:04d}"
            row = {
                'query_id': query_id,
                'filter_id': filter_id,
                'filter_selectivity': filter_selectivity,
                'k': params['k'],
                'ef_search': params['ef_search'],
                'algorithm': params['algo'],
                'm': params['m'],
                'recall': recalls[i],
                'runtime': times[i]
            }
            rows.append(row)
    elif idx_type == "ivf":
        for i in range(num_queries):
            query_id = f"q{'m' if params['query_type'] == 'movies' else 'r'}{i:04d}"
            row = {
                'query_id': query_id,
                'filter_id': filter_id,
                'filter_selectivity': filter_selectivity,
                'k': params['k'],
                'probes': params['probes'],
                'algorithm': params['algo'],
                'clusters': params['clusters'],
                'recall': recalls[i],
                'runtime': times[i]
            }
            rows.append(row)
    return rows


def main(dataset_size: str = "small", att_idx: int = 0):
    root_results = Path(f"/home/abylay/ann-benchmarks-HQ/results/ablation_seg16384/MoRe_UPD_{dataset_size}_attidx_{att_idx}")
    root_data = Path("/home/abylay/ann-benchmarks-HQ/data/datasets")
    
    result_data_hnsw = []
    result_data_ivf = []
    
    for file in root_results.rglob('*.hdf5'):
        print(f"Processing file: {file}")
        idx_type, params = parse_filename(file)
        if params is None:
            print(f"Skipping file {file} due to unmatched pattern")
            continue
        
        try:
            selectivities = load_selectivities(root_data, dataset_size, params['query_type'])
            print("Parameters: ", params)
            true_distances = 1 - load_true_distances(root_data, dataset_size, params['query_type'], params['filter_id'], params['k'])
            print(f"Number of true distances: {len(true_distances)}")
            # if file name contains "efCons" then skip
            rows = process_single_file(file, params, selectivities, true_distances, idx_type)
            if idx_type == "ivf":
                result_data_ivf.extend(rows)
            elif idx_type == "hnsw":
                result_data_hnsw.extend(rows)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            print(true_distances)
            sys.exit(1)
            continue
    
    if not result_data_hnsw and not result_data_ivf:
        print("No data processed")
        return
    
    df_hnsw = pd.DataFrame(result_data_hnsw)
    df_ivf = pd.DataFrame(result_data_ivf)

    result_file_name_hnsw = f"{root_results}/all_results_hnsw.csv"
    df_hnsw.to_csv(result_file_name_hnsw, index=False)
    print(f"Results saved to {result_file_name_hnsw}")
    print("All results DataFrame shape:", df_hnsw.shape)
    print(df_hnsw.head(3))
    
    
    result_file_name_ivf = f"{root_results}/all_results_ivf.csv"
    df_ivf.to_csv(result_file_name_ivf, index=False)
    print(f"Results saved to {result_file_name_ivf}")
    print("All results DataFrame shape:", df_ivf.shape)
    print(df_ivf.head(3))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset results")
    parser.add_argument("--dataset_size", choices=["small", "medium", "large"], default="small", help="Size of the dataset")
    parser.add_argument("--att_idx", choices=["0", "1"], default="0", help="Attribute index use")
    args = parser.parse_args()
    main(dataset_size=args.dataset_size, att_idx=args.att_idx)
