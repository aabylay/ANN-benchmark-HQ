#!/usr/bin/env python3
"""
Inner script that runs inside Docker container to execute query analysis.
This script is called from run_query_analysis.py following the runner.py pattern.
"""

import sys
import os
import argparse
import numpy as np
import h5py
import time
import json
import re
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/app/ann_benchmarks')

from ann_benchmarks.runner import (
    load_train_dataset,
    load_workload_dataset,
    load_filters,
    compile_filter
)
from ann_benchmarks.definitions import Definition, instantiate_algorithm
from ann_benchmarks.plotting.metrics import get_recall_values, knn_threshold
from ann_benchmarks.distance import metrics
from ann_benchmarks.results import build_result_filepath


def load_true_results(dataset_size: str, dataset_type: str, filter_id: int, k: int) -> tuple:
    """Load ground truth distances from HDF5 file."""
    root_data = Path("/home/app/data/datasets")
    true_path = root_data / f"MoRe_{dataset_size}" / "queries" / f"queries_flex_{dataset_type}_sim_0_{filter_id}.hdf5"
    
    print(f"Loading true results from: {true_path}", flush=True)
    with h5py.File(true_path, 'r') as f:
        true_distances = f['distances'][:, :k]
        true_neighbors = None
    
    return true_neighbors, true_distances


def calculate_result_distances(result_ids, query_vec, train_vecs, distance_metric, k):
    """Calculate distances for result IDs and sort by distance ascending."""
    if len(result_ids) == 0:
        return [], []
    result_distances = []
    valid_result_ids = []
    for idx in result_ids:
        if 0 <= idx < len(train_vecs):
            dist = metrics[distance_metric].distance(query_vec, train_vecs[idx])
            result_distances.append(dist)
            valid_result_ids.append(idx)
    result_with_dist = list(zip(valid_result_ids, result_distances))
    # result_with_dist.sort(key=lambda x: x[1])
    sorted_result_ids = [idx for idx, _ in result_with_dist[:k]]
    sorted_distances = [dist for _, dist in result_with_dist[:k]]
    return sorted_result_ids, sorted_distances


def calculate_recall_for_query(true_distances, result_distances, k):
    """Calculate recall for a single query using the same method as make_results.py.
    
    Args:
        true_distances: 1D array of true distances (length k, sorted ascending)
        result_distances: 1D array of result distances (can be any length, sorted ascending)
        k: Number of nearest neighbors
        
    Returns:
        Recall value (0.0 to 1.0)
    """
    if len(result_distances) == 0:
        return 0.0
    
    # Ensure true_distances has exactly k elements and convert to numpy array
    true_dists_k = 1 - np.array(true_distances[:k] if len(true_distances) >= k else true_distances)
    
    # Convert result_distances to numpy array (ensure it's 1D array of floats)
    result_dists_array = np.array(result_distances, dtype=float)
    
    # Reshape to 2D arrays as expected by get_recall_values:
    # - true_distances: (1, k) - exactly k distances
    # - result_distances: (1, num_results) - can have any number of results
    true_dists_2d = true_dists_k.reshape(1, -1)  # Shape: (1, k)
    result_dists_2d = result_dists_array.reshape(1, -1)  # Shape: (1, num_results)
    
    # Use the same parameters as make_results.py
    epsilon = 1e-6
    runs = get_recall_values(true_dists_2d, result_dists_2d, k, knn_threshold, epsilon=epsilon)
    
    # runs[2] contains per-query recall values, we have only one query
    recall = runs[2][0]
    return recall


def get_query_plan(algo, query_vector, k, filter_str, compiled_filter):
    """Get query plan using EXPLAIN (ANALYZE, BUFFERS, VERBOSE)."""
    if compiled_filter == ["No_filter"] or compiled_filter == "No_filter":
        query_sql = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        params = (query_vector, k)
    else:
        attr, literal, value = compiled_filter[0], compiled_filter[1], compiled_filter[2]
        query_sql = f"SELECT id FROM items WHERE filter_attr {literal} {value}::FLOAT ORDER BY embedding <=> %s LIMIT %s"
        params = (query_vector, k)
    
    explain_query = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE) " + query_sql
    try:
        algo._cur.execute(explain_query, params, binary=True, prepare=True)
        plan_rows = algo._cur.fetchall()
        plan = "\n".join([row[0] for row in plan_rows])
        return plan
    except Exception as e:
        return f"Error getting query plan: {e}"


def build_result_hdf5_path(algorithm: str, k: int, filter_id: int, ef_search: int, nprobe: int,
                           dataset_size: str, dataset_type: str, m: int, ef_construction: int,
                           clusters: int, att_idx: int = 1) -> Path:
    """Build the path to the result HDF5 file."""
    if algorithm == "pgvector":
        algo_def = Definition(
            algorithm="pgvector",
            docker_tag=None,
            module="ann_benchmarks.algorithms.pgvector",
            constructor="PGVector",
            arguments=["angular", {"M": m, "efConstruction": ef_construction}],
            query_argument_groups=[[ef_search]],
            disabled=False
        )
        query_args = [ef_search]
    elif algorithm == "pgvector_ivf":
        algo_def = Definition(
            algorithm="pgvector_ivf",
            docker_tag=None,
            module="ann_benchmarks.algorithms.pgvector_ivf",
            constructor="PGVector",
            arguments=["angular", {"clusters": clusters}],
            query_argument_groups=[[nprobe]],
            disabled=False
        )
        query_args = [nprobe]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    filepath = build_result_filepath(
        dataset_name=None,
        count=k,
        definition=algo_def,
        query_arguments=query_args,
        batch_mode=False,
        filter_id=filter_id,
        dataset_size=dataset_size,
        data_table=dataset_type,
        att_idx=att_idx
    )
    return Path(filepath)


def load_csv_recall(csv_path: Path, query_id: str, filter_id: int, k: int, 
                    algorithm: str, ef_search: int = None, nprobe: int = None,
                    m: int = None, clusters: int = None) -> tuple:
    """Load recall from CSV file for a specific query.
    
    Returns:
        Tuple of (recall, runtime) or (None, None) if not found
    """
    if not csv_path.exists():
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Match by query_id, filter_id, k, algorithm
        mask = (df['query_id'] == query_id) & (df['filter_id'] == filter_id) & (df['k'] == k) & (df['algorithm'] == algorithm)
        
        # Add algorithm-specific matching
        if algorithm == "pgvector" and ef_search is not None and m is not None:
            if 'ef_search' in df.columns:
                mask = mask & (df['ef_search'] == ef_search)
            if 'm' in df.columns:
                mask = mask & (df['m'] == m)
        elif algorithm == "pgvector_ivf" and nprobe is not None and clusters is not None:
            if 'probes' in df.columns:
                mask = mask & (df['probes'] == nprobe)
            if 'clusters' in df.columns:
                mask = mask & (df['clusters'] == clusters)
        
        matches = df[mask]
        if len(matches) > 0:
            row = matches.iloc[0]
            return row['recall'], row.get('runtime', None)
        return None, None
    except Exception as e:
        print(f"Error loading CSV recall: {e}", flush=True)
        return None, None


def load_hdf5_distances(hdf5_path: Path, query_idx: int, k: int) -> np.ndarray:
    """Load distances from HDF5 file for a specific query.
    
    Returns:
        Array of distances for the query, or None if file doesn't exist
    """
    if not hdf5_path.exists():
        return None
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            distances = f['distances'][:]
            if query_idx < len(distances):
                # Return first k distances
                return distances[query_idx][:k]
        return None
    except Exception as e:
        print(f"Error loading HDF5 distances: {e}", flush=True)
        return None


def run_analysis(algorithm: str, k: int, filter_id: int, ef_search: int, nprobe: int,
                 dataset_size: str, dataset_type: str,
                 m: int = 10, ef_construction: int = 50, clusters: int = 100, att_idx: int = 1):
    """Run query analysis for a single algorithm."""
    print(f"\n{'='*80}", flush=True)
    print(f"Running Query Analysis: {algorithm}", flush=True)
    print(f"Parameters: k={k}, filter_id={filter_id}, ef_search={ef_search}, nprobe={nprobe}", flush=True)
    print(f"Dataset: {dataset_type}, Size: {dataset_size}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Load datasets
    print("Loading datasets...", flush=True)
    train_ids, train_vecs, train_attrs, dimension = load_train_dataset(dataset_type, dataset_size)
    filter_ids, filters = load_filters(dataset_type, dataset_size)
    X_test, distance = load_workload_dataset(dataset_type, str(filter_id), dataset_size)
    
    if filter_id >= len(filters):
        raise ValueError(f"Filter ID {filter_id} out of range. Available filters: {len(filters)}")
    filter_str = filters[filter_id]
    compiled_filter = compile_filter(filter_str)
    
    print(f"Filter: {filter_str} -> {compiled_filter}", flush=True)
    print(f"Number of queries: {len(X_test)}", flush=True)
    print(f"Training vectors shape: {train_vecs.shape}", flush=True)
    print(flush=True)
    
    # Load ground truth
    true_neighbors_list, true_distances = load_true_results(dataset_size, dataset_type, filter_id, k)
    
    # Limit to first 10 queries for testing (optional - remove these lines for full run)
    X_test = X_test[:10]
    true_distances = true_distances[:10]  # Also limit true distances to match
    
    # Create algorithm definition
    if algorithm == "pgvector":
        algo_def = Definition(
            algorithm="pgvector",
            docker_tag=None,
            module="ann_benchmarks.algorithms.pgvector",
            constructor="PGVector",
            arguments=["angular", {"M": m, "efConstruction": ef_construction}],
            query_argument_groups=[[ef_search]],
            disabled=False
        )
        query_param = ef_search
    elif algorithm == "pgvector_ivf":
        algo_def = Definition(
            algorithm="pgvector_ivf",
            docker_tag=None,
            module="ann_benchmarks.algorithms.pgvector_ivf",
            constructor="PGVector",
            arguments=["angular", {"clusters": clusters}],
            query_argument_groups=[[nprobe]],
            disabled=False
        )
        query_param = nprobe
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Instantiate algorithm
    print(f"Instantiating algorithm: {algorithm}...", flush=True)
    algo = instantiate_algorithm(algo_def)
    
    # Build index
    print(f"Building {algorithm} index...", flush=True)
    algo.fit(train_ids, train_vecs, train_attrs[0] if dataset_type == "movies" else train_attrs[4], dataset_type)
    algo.fit_idx(dataset_type)
    algo.set_query_arguments(query_param)
    
    print("Index built successfully!\n", flush=True)
    
    # Build paths to result files
    root_results = Path("/home/app/results")
    csv_type = "hnsw" if algorithm == "pgvector" else "ivf"
    csv_path = root_results / f"MoRe_UPD_{dataset_size}_attidx_{att_idx}" / f"all_results_{csv_type}.csv"
    hdf5_path = build_result_hdf5_path(algorithm, k, filter_id, ef_search, nprobe,
                                       dataset_size, dataset_type, m, ef_construction,
                                       clusters, att_idx)
    
    print(f"CSV path: {csv_path}", flush=True)
    print(f"HDF5 path: {hdf5_path}", flush=True)
    print(flush=True)
    
    # Run queries
    print(f"Running {len(X_test)} queries...\n", flush=True)
    
    for query_idx in range(len(X_test)):
        query_vec = X_test[query_idx]
        # Load true distances directly from file (already sorted, matching make_results.py)
        true_dists = true_distances[query_idx]
        
        query_plan = get_query_plan(algo, query_vec, k, filter_str, compiled_filter)
        
        start_time = time.time()
        result_ids = algo.query(query_vec, k, compiled_filter)
        query_time = time.time() - start_time
        
        sorted_result_ids, sorted_result_distances = calculate_result_distances(
            result_ids, query_vec, train_vecs, distance, k
        )
        
        # Use true distances directly for recall calculation (matching make_results.py approach)
        recall = calculate_recall_for_query(true_dists, sorted_result_distances, k)
        
        # Load CSV recall and HDF5 distances
        prefix = "qm" if dataset_type == "movies" else "qr"
        query_id = f"{prefix}{query_idx:04d}"
        csv_recall, csv_runtime = load_csv_recall(csv_path, query_id, filter_id, k, algorithm,
                                                  ef_search, nprobe, m, clusters)
        hdf5_dists = load_hdf5_distances(hdf5_path, query_idx, k)
        
        print(f"\n{'─'*80}", flush=True)
        print(f"Query {query_idx + 1}/{len(X_test)}", flush=True)
        print(f"{'─'*80}", flush=True)
        print(f"Query ID: {query_id}", flush=True)
        print(f"Query Time: {query_time*1000:.2f} ms", flush=True)
        print(f"Recall (calculated): {recall:.4f}", flush=True)
        if csv_recall is not None:
            print(f"Recall (from CSV): {csv_recall:.4f}", flush=True)
        if csv_runtime is not None:
            print(f"Runtime (from CSV): {csv_runtime*1000:.4f} ms", flush=True)
        print(f"\nQuery Results (sorted by distance ascending):", flush=True)
        print(f"  IDs: {sorted_result_ids}", flush=True)
        print(f"  Distances: {[f'{1 - d:.6f}' for d in sorted_result_distances]}", flush=True)
        if hdf5_dists is not None:
            print(f"\nHDF5 Results (from file):", flush=True)
            print(f"  Distances: {[f'{d:.6f}' for d in hdf5_dists]}", flush=True)
        print(f"\nTrue Results (sorted by distance ascending):", flush=True)
        print(f"  Distances: {[f'{d:.6f}' for d in true_dists]}", flush=True)
        print(f"\nQuery Plan:\n{query_plan}", flush=True)
        print(f"{'─'*80}\n", flush=True)
    
    print(f"\nFinished processing all queries for {algorithm}\n", flush=True)
    
    # Cleanup
    try:
        algo.done()
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run query analysis inside Docker container")
    parser.add_argument("--algorithm", default="pgvector_ivf", type=str, choices=["pgvector", "pgvector_ivf"],
                       help="Algorithm to run")
    parser.add_argument("--k", default=10, type=int, help="Number of nearest neighbors")
    parser.add_argument("--filter_id", default=1, type=int, help="Filter ID")
    parser.add_argument("--ef_search", default=1000, type=int, help="ef_search parameter for pgvector (HNSW)")
    parser.add_argument("--nprobe", default=100, type=int, help="nprobe parameter for pgvector_ivf")
    parser.add_argument("--dataset_size", default="small", type=str, choices=["small", "medium", "large"], 
                       help="Dataset size")
    parser.add_argument("--dataset_type", default="movies", type=str, choices=["movies", "reviews"], 
                       help="Dataset type")
    parser.add_argument("--m", type=int, default=10, help="M parameter for HNSW (default: 10)")
    parser.add_argument("--ef_construction", default=50, type=int, 
                       help="ef_construction for HNSW (default: 50)")
    parser.add_argument("--clusters", type=int, default=100, 
                       help="Number of clusters for IVF (default: 100)")
    parser.add_argument("--att_idx", type=int, default=1, choices=[0, 1],
                       help="Attribute index (0 or 1, default: 1)")
    
    args = parser.parse_args()
    
    run_analysis(
        algorithm=args.algorithm,
        k=args.k,
        filter_id=args.filter_id,
        ef_search=args.ef_search,
        nprobe=args.nprobe,
        dataset_size=args.dataset_size,
        dataset_type=args.dataset_type,
        m=args.m,
        ef_construction=args.ef_construction,
        clusters=args.clusters,
        att_idx=args.att_idx
    )


if __name__ == "__main__":
    main()

