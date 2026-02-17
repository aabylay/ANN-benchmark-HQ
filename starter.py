import argparse
import os
import re
from make_yaml import make_yaml

ef_s_list = [100, 200, 500, 1000]


def write_milvus_user_yaml(segment_size_mb=16384):
    """Write Milvus user.yaml to <project>/milvus_data/ on the HOST.

    Snap Docker has a private /tmp namespace and cannot see files in the host's
    /tmp.  We write to the project directory instead (under /home/...) which
    snap Docker CAN access.  The benchmark container's start_milvus() rewrites
    the docker-compose.yml volume paths from /tmp/ to this directory at runtime.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    milvus_data_dir = os.path.join(project_dir, "milvus_data")
    os.makedirs(milvus_data_dir, exist_ok=True)

    disk_segment_size = segment_size_mb * 2
    user_yaml_content = (
        f"# Milvus config written by starter.py (segment size: {segment_size_mb} MB)\n"
        f"dataCoord:\n"
        f"  segment:\n"
        f"    maxSize: {segment_size_mb}\n"
        f"    diskSegmentMaxSize: {disk_segment_size}\n"
        f"    sealProportion: 0.12\n"
    )
    yaml_path = os.path.join(milvus_data_dir, "user.yaml")
    with open(yaml_path, "w") as f:
        f.write(user_yaml_content)
    print(f"[starter] Wrote {yaml_path} with segment maxSize={segment_size_mb}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running ANN-Benchmark with filters.")
    parser.add_argument("--dataset_size", type=str, default="small", help="Size of the dataset (small, medium, large).")
    args = parser.parse_args()
    dataset_size = args.dataset_size

    # Extract segment size from the config's docker_tag and write user.yaml on the HOST
    # so Milvus containers can mount it correctly.
    # Default to 16384 MB; adjust if using a different docker tag.
    write_milvus_user_yaml(segment_size_mb=16384)

    # TODO: NEW ONE TO BE IMPLEMENTED "hnsw_cagra(faiss)"    
    for dataset_size in ["large", "small", "medium"]: # "small", "medium", "large"
        print(f"\n\n=======================================CHECKING DATASET SIZE {dataset_size}=======================================\n\n", flush=True)

        """ Available algorithms list: 
            ["milvus-hnsw", "pgvector", "hnsw(faiss)"]
            ["milvus-ivfflat", "pgvector_ivf", "faiss-ivf"]
        """
        
        for algo in ["milvus-ivfflat"]: # "pgvector_ivf", "milvus-ivfflat", "faiss-ivf", "milvus-hnsw", "pgvector", "hnsw(faiss)"
            print(f"----------------------------------------\nRunning experiments for algorithm: {algo}", flush=True)            
            if algo in ["milvus-hnsw", "pgvector", "hnsw(faiss)"]:
                for m in [5]: # [5, 10, 15]
                    ef_c = m * 5
                    make_yaml(algo, m, ef_c, ef_s_list)
                    print(f"Running for HNSW on dataset size '{dataset_size}' with m: {m}", flush=True)
                    os.system(f"python run.py --algorithm \"{algo}\" --dataset glove-100-angular --dataset_size {dataset_size}")
                    print(f"Finished for HNSW on dataset size '{dataset_size}' with m: {m}, algo: {algo}\n", flush=True)
            
            if algo in ["pgvector_ivf", "faiss-ivf", "milvus-ivfflat"]:
                make_yaml(algo, 0, None, None, True, f"{dataset_size}")
                print(f"Running for IVF on dataset size '{dataset_size}'...", flush=True)
                #### To pass placeholder for clusters and change later in runner
                os.system(f"python run.py --algorithm \"{algo}\" --dataset glove-100-angular --dataset_size {dataset_size}")
                print(f"Finished for IVF on dataset size '{dataset_size}', algo: {algo}\n", flush=True)
                
            print(f"----------------------------------------\nFinished experiments for algorithm: {algo}", flush=True)