"""
Starter script for Milvus segment size ablation study.

This script orchestrates running benchmarks with different segment sizes,
using run_ablation.py which stores results separately for each segment size.

Results are stored in: results/ablation_seg{segment_size}/...
"""
import argparse
import os
from datetime import datetime
from make_yaml_ablation import make_yaml_ablation

# Segment sizes for ablation study (in MB)
SEGMENT_SIZES = [512, 1024, 2048, 4096, 8192, 16384]

# HNSW parameters
ef_s_list = [100, 200, 500, 1000]


def run_ablation_study(dataset_size="small", algorithms=None, segment_sizes=None):
    """
    Run ablation study on Milvus with different segment sizes.
    
    Args:
        dataset_size: Size of the dataset (small, medium, large)
        algorithms: List of algorithms to test (default: ["milvus-hnsw"])
        segment_sizes: List of segment sizes to test (default: all)
    """
    if algorithms is None:
        algorithms = ["milvus-hnsw"]
    
    if segment_sizes is None:
        segment_sizes = SEGMENT_SIZES
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("MILVUS SEGMENT SIZE ABLATION STUDY")
    print("=" * 70)
    print(f"Started at: {timestamp}")
    print(f"Dataset size: {dataset_size}")
    print(f"Algorithms: {algorithms}")
    print(f"Segment sizes: {segment_sizes} MB")
    print(f"Results will be stored in: results/ablation_seg{{size}}/...")
    print("=" * 70)
    
    for seg_size in segment_sizes:
        print(f"\n{'='*70}")
        print(f"TESTING SEGMENT SIZE: {seg_size} MB")
        print(f"Using Docker image: ann-benchmarks-milvus-seg{seg_size}")
        print(f"{'='*70}\n")
        
        for algo in algorithms:
            print(f"\n--- Running {algo} with segment size {seg_size} MB ---\n")
            
            if algo == "milvus-hnsw":
                for m in [15]:  # Can extend to [5, 10, 15] for full sweep
                    ef_c = m * 5
                    
                    # Generate config with correct docker tag for this segment size
                    docker_tag = make_yaml_ablation(
                        segment_size=seg_size,
                        algo=algo,
                        m=m,
                        ef_c=ef_c,
                        ef_s_list=ef_s_list,
                        ivf_algo=False,
                        dataset_size=dataset_size
                    )
                    
                    print(f"Running {algo} with M={m}, efConstruction={ef_c}, segment_size={seg_size}MB")
                    print(f"Docker image: {docker_tag}")
                    
                    # Use run_ablation.py with segment_size argument
                    cmd = (f'python run_ablation.py '
                           f'--algorithm "{algo}" '
                           f'--dataset glove-100-angular '
                           f'--dataset_size {dataset_size} '
                           f'--segment_size {seg_size}')
                    
                    print(f"Executing: {cmd}")
                    exit_code = os.system(cmd)
                    
                    if exit_code != 0:
                        print(f"WARNING: Command exited with code {exit_code}")
                    
                    print(f"Completed: {algo} M={m}, segment_size={seg_size}MB\n")
            
            elif algo == "milvus-ivfflat":
                # Generate config for IVF
                docker_tag = make_yaml_ablation(
                    segment_size=seg_size,
                    algo=algo,
                    ivf_algo=True,
                    dataset_size=dataset_size
                )
                
                print(f"Running {algo} with segment_size={seg_size}MB")
                print(f"Docker image: {docker_tag}")
                
                cmd = (f'python run_ablation.py '
                       f'--algorithm "{algo}" '
                       f'--dataset glove-100-angular '
                       f'--dataset_size {dataset_size} '
                       f'--segment_size {seg_size}')
                
                print(f"Executing: {cmd}")
                exit_code = os.system(cmd)
                
                if exit_code != 0:
                    print(f"WARNING: Command exited with code {exit_code}")
                
                print(f"Completed: {algo}, segment_size={seg_size}MB\n")
            
            elif algo == "milvus-scann":
                # Generate config for SCANN
                docker_tag = make_yaml_ablation(
                    segment_size=seg_size,
                    algo=algo,
                    ivf_algo=False,
                    dataset_size=dataset_size
                )
                
                print(f"Running {algo} with segment_size={seg_size}MB")
                print(f"Docker image: {docker_tag}")
                
                cmd = (f'python run_ablation.py '
                       f'--algorithm "{algo}" '
                       f'--dataset glove-100-angular '
                       f'--dataset_size {dataset_size} '
                       f'--segment_size {seg_size}')
                
                print(f"Executing: {cmd}")
                exit_code = os.system(cmd)
                
                if exit_code != 0:
                    print(f"WARNING: Command exited with code {exit_code}")
                
                print(f"Completed: {algo}, segment_size={seg_size}MB\n")
        
        print(f"\nCompleted all algorithms for segment size {seg_size}MB")
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"Results are stored in separate directories:")
    for seg_size in segment_sizes:
        print(f"  - results/ablation_seg{seg_size}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milvus segment size ablation study")
    parser.add_argument("--dataset_size", type=str, default="small",
                        help="Size of the dataset (small, medium, large)")
    parser.add_argument("--algorithms", type=str, nargs="+", 
                        default=["milvus-hnsw"],
                        help="List of algorithms to test (milvus-hnsw, milvus-ivfflat, milvus-scann)")
    parser.add_argument("--segment_sizes", type=int, nargs="+",
                        default=None,
                        help="Segment sizes to test in MB (default: 512 1024 2048 4096 8192 16384)")
    
    args = parser.parse_args()
    
    run_ablation_study(
        dataset_size=args.dataset_size,
        algorithms=args.algorithms,
        segment_sizes=args.segment_sizes
    )
