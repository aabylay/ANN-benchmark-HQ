import argparse
import os
from make_yaml import make_yaml

# HNSW efSearch sweep (finer grid for Direction 4.5: separate model error from
# grid-coarseness). starter.py regenerates each HNSW config.yml from this list
# via make_yaml(), so this is the single source of truth for the efSearch grid.
ef_s_list = [40, 60, 80]

# Full FANNS plan set used for hard/superhard packs.
HARD_ALGOS = [
    "faiss-flat",
    "hnsw(faiss)",
    "hnsw(faiss)-post",
    "faiss-ivf",
    "faiss-ivf-post",
    "pgvector_bf",
    "pgvector",
    "pgvector_ivf",
]

# Flex large-MoRe campaign keeps the historical FAISS-only default list.
FLEX_ALGOS = [
    "faiss-flat",
    "hnsw(faiss)",
    "hnsw(faiss)-post",
    "faiss-ivf",
    "faiss-ivf-post",
]


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


def run_algo(algo: str, dataset_size: str, workload: str) -> None:
    # Always pass --workload explicitly so Docker path is unambiguous.
    workload_flag = f" --workload {workload}"

    if algo in ["milvus-hnsw", "pgvector", "hnsw(faiss)", "hnsw(faiss)-post"]:
        make_yaml(algo, 16, 128, ef_s_list)
        print(
            f"Running HNSW '{algo}' on dataset size '{dataset_size}' "
            f"workload={workload} (m=16, ef_construction=128)",
            flush=True,
        )
        os.system(
            f'python run.py --algorithm "{algo}" --dataset glove-100-angular '
            f"--dataset_size {dataset_size}{workload_flag}"
        )
        print(f"Finished HNSW '{algo}' on dataset size '{dataset_size}'\n", flush=True)

    elif algo in ["pgvector_ivf", "faiss-ivf", "faiss-ivf-post", "milvus-ivfflat"]:
        make_yaml(algo, 0, None, None, True, f"{dataset_size}")
        print(
            f"Running IVFFlat '{algo}' on dataset size '{dataset_size}' "
            f"workload={workload} (clusters ~ sqrt(|D|))...",
            flush=True,
        )
        os.system(
            f'python run.py --algorithm "{algo}" --dataset glove-100-angular '
            f"--dataset_size {dataset_size}{workload_flag}"
        )
        print(f"Finished IVFFlat '{algo}' on dataset size '{dataset_size}'\n", flush=True)

    elif algo in ["faiss-flat", "pgvector_bf"]:
        make_yaml(algo)
        print(
            f"Running brute-force '{algo}' on dataset size '{dataset_size}' "
            f"workload={workload}...",
            flush=True,
        )
        os.system(
            f'python run.py --algorithm "{algo}" --dataset glove-100-angular '
            f"--dataset_size {dataset_size}{workload_flag}"
        )
        print(f"Finished brute-force '{algo}' on dataset size '{dataset_size}'\n", flush=True)

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running ANN-Benchmark with filters.")
    parser.add_argument(
        "--dataset_size",
        type=str,
        default="small",
        help="Size of the dataset (small, medium, large).",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="flex",
        choices=["flex", "hard", "superhard", "hard_superhard"],
        help="Workload mode. Default flex preserves the existing large-MoRe campaign.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Comma-separated algorithm names (default depends on workload).",
    )
    args = parser.parse_args()

    write_milvus_user_yaml(segment_size_mb=16384)

    HNSW_M = 16
    HNSW_EF_C = 128

    if args.workload == "hard_superhard":
        workloads = ["hard", "superhard"]
    else:
        workloads = [args.workload]

    if args.algorithms:
        algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    elif args.workload == "flex":
        algos = list(FLEX_ALGOS)
    else:
        algos = list(HARD_ALGOS)

    for dataset_size in ["large"]:
        print(
            f"\n\n=======================================CHECKING DATASET SIZE {dataset_size}"
            f"=======================================\n\n",
            flush=True,
        )
        for workload in workloads:
            print(
                f"\n===== WORKLOAD {workload} | algos={algos} =====\n",
                flush=True,
            )
            for algo in algos:
                print(
                    f"----------------------------------------\n"
                    f"Running experiments for algorithm: {algo} (workload={workload})",
                    flush=True,
                )
                run_algo(algo, dataset_size, workload)
                print(
                    f"----------------------------------------\n"
                    f"Finished experiments for algorithm: {algo}",
                    flush=True,
                )
