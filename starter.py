import argparse
import json
import os
import sys
from pathlib import Path

import h5py

from ann_benchmarks.corpora import (
    DEFAULT_QO_SAMPLE_FRAC,
    DEFAULT_SAMPLE_SEED,
    build_sample_manifest,
    dump_manifest_bytes,
    pack_root,
    validate_qo_sample_frac,
)
from make_yaml import make_yaml

# HNSW efSearch sweep (finer grid for Direction 4.5: separate model error from
# grid-coarseness). starter.py regenerates each HNSW config.yml from this list
# via make_yaml(), so this is the single source of truth for the efSearch grid.
# 17 points, matching the large IVF probe grid in cardinality/coverage so HNSW
# optimal-plan figures get comparable HP resolution. Construction stays fixed
# (M=16, efConstruction=128).
ef_s_list = [
    10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    250, 300, 400, 500, 600, 800,
]

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


def _ensure_caselaw_manifest_frac(frac: float, max_queries: int | None) -> None:
    """Update sample_manifest.json when --qo-sample-frac differs (unless --max-queries is set)."""
    if max_queries is not None:
        return
    root = Path(pack_root("caselaw"))
    manifest_path = root / "sample_manifest.json"
    native_path = root / "native" / "cases_native.hdf5"
    if not manifest_path.is_file() or not native_path.is_file():
        return
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    if abs(float(current.get("frac", DEFAULT_QO_SAMPLE_FRAC)) - float(frac)) < 1e-12:
        return
    with h5py.File(native_path, "r") as f:
        sels = f["selectivity"][:]
        nres = f["nres"][:]
        if hasattr(f["filter"], "asstr"):
            filters = list(f["filter"].asstr()[:])
        else:
            filters = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["filter"][:]
            ]
        n_base = int(f.attrs.get("N", current.get("N", 7_414_023)))
    updated = build_sample_manifest(
        sels, nres, filters, frac=frac, seed=DEFAULT_SAMPLE_SEED, n_base=n_base
    )
    manifest_path.write_bytes(dump_manifest_bytes(updated))


def run_algo(
    algo: str,
    dataset_size: str,
    workload: str,
    corpus: str = "morevec",
    qo_sample_frac: float = DEFAULT_QO_SAMPLE_FRAC,
    max_queries: int | None = None,
) -> None:
    # Always pass --workload explicitly so Docker path is unambiguous.
    extra_flags = f" --workload {workload}"
    if corpus == "caselaw":
        extra_flags += f" --corpus caselaw --qo-sample-frac {qo_sample_frac}"
        if max_queries is not None:
            extra_flags += f" --max-queries {int(max_queries)}"

    py = sys.executable or "python"

    if algo in ["milvus-hnsw", "pgvector", "hnsw(faiss)", "hnsw(faiss)-post"]:
        make_yaml(algo, 16, 128, ef_s_list)
        print(
            f"Running HNSW '{algo}' on dataset size '{dataset_size}' "
            f"corpus={corpus} workload={workload} (m=16, ef_construction=128)",
            flush=True,
        )
        rc = os.system(
            f'"{py}" run.py --algorithm "{algo}" --dataset glove-100-angular '
            f"--dataset_size {dataset_size}{extra_flags}"
        )
        if rc != 0:
            raise RuntimeError(f"run.py failed with exit code {rc} for {algo}")
        print(f"Finished HNSW '{algo}' on dataset size '{dataset_size}'\n", flush=True)

    elif algo in ["pgvector_ivf", "faiss-ivf", "faiss-ivf-post", "milvus-ivfflat"]:
        make_yaml(algo, 0, None, None, True, f"{dataset_size}")
        print(
            f"Running IVFFlat '{algo}' on dataset size '{dataset_size}' "
            f"corpus={corpus} workload={workload} (clusters ~ sqrt(|D|))...",
            flush=True,
        )
        rc = os.system(
            f'"{py}" run.py --algorithm "{algo}" --dataset glove-100-angular '
            f"--dataset_size {dataset_size}{extra_flags}"
        )
        if rc != 0:
            raise RuntimeError(f"run.py failed with exit code {rc} for {algo}")
        print(f"Finished IVFFlat '{algo}' on dataset size '{dataset_size}'\n", flush=True)

    elif algo in ["faiss-flat", "pgvector_bf"]:
        make_yaml(algo)
        print(
            f"Running brute-force '{algo}' on dataset size '{dataset_size}' "
            f"corpus={corpus} workload={workload}...",
            flush=True,
        )
        rc = os.system(
            f'"{py}" run.py --algorithm "{algo}" --dataset glove-100-angular '
            f"--dataset_size {dataset_size}{extra_flags}"
        )
        if rc != 0:
            raise RuntimeError(f"run.py failed with exit code {rc} for {algo}")
        print(f"Finished brute-force '{algo}' on dataset size '{dataset_size}'\n", flush=True)

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running ANN-Benchmark with filters.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["morevec", "caselaw", "both"],
        help="Corpus to run: morevec, caselaw, or both (default: both).",
    )
    parser.add_argument(
        "--qo-sample-frac",
        type=float,
        default=DEFAULT_QO_SAMPLE_FRAC,
        help="Stratified training sample fraction for CaseLaw in [0.05, 0.20].",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional query prefix truncation for CaseLaw smoke runs.",
    )
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

    validate_qo_sample_frac(args.qo_sample_frac)
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be >= 1")

    write_milvus_user_yaml(segment_size_mb=16384)

    HNSW_M = 16
    HNSW_EF_C = 128

    run_morevec = args.dataset in ("morevec", "both")
    run_caselaw = args.dataset in ("caselaw", "both")

    if run_morevec:
        if (
            args.dataset == "morevec"
            and (
                abs(args.qo_sample_frac - DEFAULT_QO_SAMPLE_FRAC) > 1e-12
                or args.max_queries is not None
            )
        ):
            print(
                "[starter] Notice: --qo-sample-frac and --max-queries apply to CaseLaw only and are ignored for morevec.",
                flush=True,
            )

        if args.workload == "hard_superhard":
            workloads = ["hard", "superhard"]
        else:
            workloads = [args.workload]

        if args.algorithms:
            more_algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
        elif args.workload == "flex":
            more_algos = list(FLEX_ALGOS)
        else:
            more_algos = list(HARD_ALGOS)

        for dataset_size in ["large"]:
            print(
                f"\n\n=======================================CHECKING DATASET SIZE {dataset_size}"
                f"=======================================\n\n",
                flush=True,
            )
            for workload in workloads:
                print(
                    f"\n===== WORKLOAD {workload} | algos={more_algos} =====\n",
                    flush=True,
                )
                for algo in more_algos:
                    print(
                        f"----------------------------------------\n"
                        f"Running experiments for algorithm: {algo} (workload={workload})",
                        flush=True,
                    )
                    run_algo(algo, dataset_size, workload, corpus="morevec")
                    print(
                        f"----------------------------------------\n"
                        f"Finished experiments for algorithm: {algo}",
                        flush=True,
                    )

    if run_caselaw:
        if args.workload in ("hard", "superhard", "hard_superhard"):
            print(
                f"Warning: CaseLaw has no HCBGen workload ({args.workload}); running CaseLaw native workload.",
                flush=True,
            )
        if args.algorithms:
            caselaw_algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
        else:
            caselaw_algos = list(FLEX_ALGOS)

        _ensure_caselaw_manifest_frac(args.qo_sample_frac, args.max_queries)

        for dataset_size in ["large"]:
            print(
                f"\n===== CASELAW NATIVE WORKLOAD | frac={args.qo_sample_frac} | algos={caselaw_algos} =====\n",
                flush=True,
            )
            for algo in caselaw_algos:
                print(
                    f"----------------------------------------\n"
                    f"Running CaseLaw experiments for algorithm: {algo} (workload=native)",
                    flush=True,
                )
                run_algo(
                    algo,
                    dataset_size,
                    "native",
                    corpus="caselaw",
                    qo_sample_frac=args.qo_sample_frac,
                    max_queries=args.max_queries,
                )
                print(
                    f"----------------------------------------\n"
                    f"Finished CaseLaw experiments for algorithm: {algo}",
                    flush=True,
                )
