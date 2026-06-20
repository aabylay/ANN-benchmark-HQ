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

    # Fixed index construction params for the FANNS sweep.
    HNSW_M = 16          # HNSW graph degree
    HNSW_EF_C = 64       # HNSW efConstruction
    # IVF number of lists is FIXED at ~sqrt(|D|) and computed per table inside
    # each algorithm's fit() (passed as clusters=0 -> auto).

    for dataset_size in ["large"]: # only the large MoRe dataset is benchmarked
        print(f"\n\n=======================================CHECKING DATASET SIZE {dataset_size}=======================================\n\n", flush=True)

        """ FANNS plans benchmarked (2 systems x 3 index types):
            FAISS    : faiss-flat (brute-force, IndexFlatIP + bitset pre-filter)
                       hnsw(faiss) (HNSW, bitset pre-filter)
                       faiss-ivf  (IVFFlat, bitset pre-filter)
            PG-Vector: pgvector_bf (brute-force, no index, SQL post-filter)
                       pgvector    (HNSW, post-filter w/ iterative scan)
                       pgvector_ivf(IVFFlat, post-filter w/ iterative scan)
        """

        for algo in ["faiss-flat", "hnsw(faiss)", "faiss-ivf",
                     "pgvector_bf", "pgvector", "pgvector_ivf"]:
            print(f"----------------------------------------\nRunning experiments for algorithm: {algo}", flush=True)

            if algo in ["milvus-hnsw", "pgvector", "hnsw(faiss)"]:  # HNSW plans (fixed m, ef_construction)
                make_yaml(algo, HNSW_M, HNSW_EF_C, ef_s_list)
                print(f"Running HNSW '{algo}' on dataset size '{dataset_size}' (m={HNSW_M}, ef_construction={HNSW_EF_C})", flush=True)
                os.system(f"python run.py --algorithm \"{algo}\" --dataset glove-100-angular --dataset_size {dataset_size}")
                print(f"Finished HNSW '{algo}' on dataset size '{dataset_size}'\n", flush=True)

            elif algo in ["pgvector_ivf", "faiss-ivf", "milvus-ivfflat"]:  # IVFFlat plans (clusters auto ~ sqrt|D|)
                make_yaml(algo, 0, None, None, True, f"{dataset_size}")
                print(f"Running IVFFlat '{algo}' on dataset size '{dataset_size}' (clusters ~ sqrt(|D|))...", flush=True)
                os.system(f"python run.py --algorithm \"{algo}\" --dataset glove-100-angular --dataset_size {dataset_size}")
                print(f"Finished IVFFlat '{algo}' on dataset size '{dataset_size}'\n", flush=True)

            elif algo in ["faiss-flat", "pgvector_bf"]:  # brute-force (exact) plans
                make_yaml(algo)
                print(f"Running brute-force '{algo}' on dataset size '{dataset_size}'...", flush=True)
                os.system(f"python run.py --algorithm \"{algo}\" --dataset glove-100-angular --dataset_size {dataset_size}")
                print(f"Finished brute-force '{algo}' on dataset size '{dataset_size}'\n", flush=True)

            print(f"----------------------------------------\nFinished experiments for algorithm: {algo}", flush=True)