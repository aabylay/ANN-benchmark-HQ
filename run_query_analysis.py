#!/usr/bin/env python3
"""
Script to run queries on pgvector and pgvector_ivf and output detailed results.

Input: k, filter_id, efSearch (for pgvector), nprobe (for pgvector_ivf), dataset_size, dataset_type
Output: For each of 100 queries, prints:
    - Query ID
    - Results (returned IDs)
    - Recall
    - Query plan
    - True results (ground truth)

This script always runs in a Docker container, following the exact pattern from runner.py.
Runs HNSW and IVF separately (not simultaneously).
"""

import argparse
import sys
import os
import docker
import psutil
import threading
import logging

# Add parent directory to path to import ann_benchmarks modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from ann_benchmarks.runner import colors


def run_docker_container(
    docker_tag: str,
    algorithm: str,
    k: int,
    filter_id: int,
    ef_search: int,
    nprobe: int,
    dataset_size: str,
    dataset_type: str,
    m: int = 10,
    ef_construction: int = 50,
    clusters: int = 100,
    att_idx: int = 1,
    cpu_limit: str = "0",
    timeout: int = None
) -> None:
    """Run query analysis in Docker container following runner.py pattern exactly.
    
    Args:
        docker_tag: Docker image tag
        algorithm: Algorithm name ("pgvector" or "pgvector_ivf")
        k: Number of nearest neighbors
        filter_id: Filter ID
        ef_search: ef_search parameter for pgvector (HNSW)
        nprobe: nprobe parameter for pgvector_ivf
        dataset_size: Dataset size (small, medium, large)
        dataset_type: Dataset type (movies, reviews)
        m: M parameter for HNSW (default 10)
        ef_construction: ef_construction for HNSW (default 50)
        clusters: Number of clusters for IVF (default 100)
        cpu_limit: CPU limit (default "0")
        timeout: Timeout in seconds (default None)
    """
    # Build command exactly like runner.py does
    cmd = [
        "--algorithm", algorithm,
        "--k", str(k),
        "--filter_id", str(filter_id),
        "--ef_search", str(ef_search),
        "--nprobe", str(nprobe),
        "--dataset_size", dataset_size,
        "--dataset_type", dataset_type,
        "--m", str(m),
        "--ef_construction", str(ef_construction),
        "--clusters", str(clusters),
        "--att_idx", str(att_idx),
    ]
    
    client = docker.from_env()
    mem_limit = psutil.virtual_memory().available
    
    print(f"Running {algorithm} with CPU limit: {cpu_limit}, memory limit: {mem_limit / (1024**3):.2f} GB", flush=True)
    
    # Run container exactly like runner.py
    container = client.containers.run(
        docker_tag,
        cmd,
        entrypoint=["python3", "-u", "/home/app/run_query_analysis_inner.py"],  # Override entrypoint to run our script
        volumes={
            os.path.abspath("/var/run/docker.sock"): {"bind": "/var/run/docker.sock", "mode": "rw"},
            os.path.abspath("ann_benchmarks"): {"bind": "/home/app/ann_benchmarks", "mode": "ro"},
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
            os.path.abspath("run_query_analysis_inner.py"): {"bind": "/home/app/run_query_analysis_inner.py", "mode": "ro"},
        },
        network_mode="host",
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        detach=True,
    )
    
    logger = logging.getLogger(f"annb.{container.short_id}")
    logger.info(
        f"Created container {container.short_id}: CPU limit {cpu_limit}, mem limit {mem_limit}, timeout {timeout}, command {cmd}"
    )
    
    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))
            # Also print to stdout
            print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
    
    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()
    
    try:
        return_value = container.wait(timeout=timeout)
        
        # Handle return value like runner.py
        if isinstance(return_value, dict):
            exit_code = return_value["StatusCode"]
            error_msg = return_value.get("Error", "")
        else:
            exit_code = return_value
            error_msg = ""
        
        if exit_code not in [0, None]:
            for line in container.logs(stream=True):
                logger.error(colors.color(line.decode(), fg="red"))
                print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
            logger.error(f"Container {container.short_id} returned exit code {exit_code} with message {error_msg}")
            raise RuntimeError(f"Container exited with code {exit_code}")
        else:
            logger.info(f"Container {container.short_id} returned exit code {exit_code}")
    except Exception as e:
        logger.error(f"Container.wait for container {container.short_id} failed with exception")
        logger.error(str(e))
        raise
    finally:
        logger.info("Removing container")
        container.remove(force=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run query analysis on pgvector and pgvector_ivf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_query_analysis.py --k 10 --filter_id 0 --ef_search 100 --nprobe 10 --dataset_size small --dataset_type movies
        """
    )
    
    parser.add_argument("--k", type=int, required=True, help="Number of nearest neighbors")
    parser.add_argument("--filter_id", type=int, required=True, help="Filter ID")
    parser.add_argument("--ef_search", type=int, required=True, help="ef_search parameter for pgvector (HNSW)")
    parser.add_argument("--nprobe", type=int, required=True, help="nprobe parameter for pgvector_ivf")
    parser.add_argument("--dataset_size", type=str, choices=["small", "medium", "large"], 
                       required=True, help="Dataset size")
    parser.add_argument("--dataset_type", type=str, choices=["movies", "reviews"], 
                       required=True, help="Dataset type")
    parser.add_argument("--m", type=int, default=10, help="M parameter for HNSW (default: 10)")
    parser.add_argument("--ef_construction", type=int, default=50, 
                       help="ef_construction for HNSW (default: 50)")
    parser.add_argument("--clusters", type=int, default=100, 
                       help="Number of clusters for IVF (default: 100)")
    parser.add_argument("--att_idx", type=int, default=1, choices=[0, 1],
                       help="Attribute index (0 or 1, default: 1)")
    parser.add_argument("--cpu_limit", type=str, default="0", 
                       help="CPU limit (default: 0)")
    parser.add_argument("--timeout", type=int, default=None,
                       help="Timeout in seconds (default: None)")
    
    args = parser.parse_args()
    
    docker_tag = "ann-benchmarks-pgvector"  # Both pgvector and pgvector_ivf use this tag
    
    # Run HNSW (pgvector) first
    print(f"\n{'='*80}", flush=True)
    print(f"Running pgvector (HNSW) analysis", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    try:
        run_docker_container(
            docker_tag=docker_tag,
            algorithm="pgvector",
            k=args.k,
            filter_id=args.filter_id,
            ef_search=args.ef_search,
            nprobe=args.nprobe,
            dataset_size=args.dataset_size,
            dataset_type=args.dataset_type,
            m=args.m,
            ef_construction=args.ef_construction,
            clusters=args.clusters,
            att_idx=args.att_idx,
            cpu_limit=args.cpu_limit,
            timeout=args.timeout
        )
    except Exception as e:
        print(f"Error running pgvector: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run IVF (pgvector_ivf) separately
    print(f"\n{'='*80}", flush=True)
    print(f"Running pgvector_ivf (IVF) analysis", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    try:
        run_docker_container(
            docker_tag=docker_tag,
            algorithm="pgvector_ivf",
            k=args.k,
            filter_id=args.filter_id,
            ef_search=args.ef_search,
            nprobe=args.nprobe,
            dataset_size=args.dataset_size,
            dataset_type=args.dataset_type,
            m=args.m,
            ef_construction=args.ef_construction,
            clusters=args.clusters,
            att_idx=args.att_idx,
            cpu_limit=args.cpu_limit,
            timeout=args.timeout
        )
    except Exception as e:
        print(f"Error running pgvector_ivf: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n{'='*80}", flush=True)
    print(f"All analyses completed successfully!", flush=True)
    print(f"{'='*80}\n", flush=True)


if __name__ == "__main__":
    main()
