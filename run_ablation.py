"""
Entry point for running Milvus segment size ablation study.

This script is analogous to run.py but uses the ablation-specific
runner and results storage that include segment_size in the path.

Usage:
    python run_ablation.py --algorithm milvus-hnsw --segment_size 512 --dataset glove-100-angular --dataset_size small
"""
from multiprocessing import freeze_support

from ann_benchmarks.main_ablation import main

if __name__ == "__main__":
    freeze_support()
    main()
