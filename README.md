# ANN-Benchmarks HQ — Filtered Approximate Nearest Neighbor Search

This repository extends [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) for **filtered ANN** — approximate nearest neighbor search with attribute filters (e.g., filtering by rating, genre, year). Benchmarks evaluate recall, query latency, and throughput under varying filter selectivity.

## Overview

- **Custom dataset**: MoRe (Movies & Reviews) — movies and reviews with embeddings and filterable attributes.
- **Tested algorithms**:
  - **pgvector**: HNSW and IVFFlat
  - **FAISS**: HNSW (`hnsw(faiss)`) and IVF (`faiss-ivf`)
  - **Milvus**: HNSW (`milvus-hnsw`) and IVFFlat (`milvus-ivfflat`); also supports IVFSQ8, IVFPQ, SCANN (check `config.yml` for enabled indexes)

## Requirements

- Python 3.8+
- Docker
- Conda (recommended for environment management)

## Setup

### 1. Conda environment

```bash
conda create -n ann-hq python=3.10
conda activate ann-hq
pip install -r requirements.txt
```

### 2. Dataset

The benchmark uses a custom dataset in HDF5 format under `data/datasets/`:

```
data/datasets/MoRe_{size}/
├── datasets/           # Train embeddings (movies, reviews)
│   ├── movies_dataset_0.hdf5
│   └── reviews_dataset_0.hdf5
├── filters/            # Filter definitions and selectivities
│   ├── movies_filters_0.hdf5
│   └── reviews_filters_0.hdf5
└── queries/            # Query workloads per filter
    └── queries_flex_{type}_sim_0_{filter_id}.hdf5
```

Supported sizes: `small`, `medium`, `large`. To obtain the MoRe dataset, please contact [aabylay@gmail.com](mailto:aabylay@gmail.com).

### 3. Docker images

Build the required Docker images:

- **Milvus**: `ann-benchmarks-milvus-seg16384` (default 16 GB segment size)
- **pgvector**: `ann-benchmarks-pgvector`
- **FAISS**: `custom-hnsw-faiss`

Example (Milvus):

```bash
docker build -t ann-benchmarks-milvus-seg16384 ann_benchmarks/algorithms/milvus/
```

For the Milvus segment-size ablation study (optional):

```bash
./build_milvus_ablation.sh
```

This builds `ann-benchmarks-milvus-seg{512,1024,2048,4096,8192,16384}`.

### 4. Milvus configuration

For Milvus, `starter.py` writes `milvus_data/user.yaml` with segment settings. Ensure Docker can access the project directory (e.g., avoid snap Docker with a private `/tmp`).

---

## Running benchmarks

### Main runner

Standard benchmark entry point:

```bash
python run.py --algorithm <ALGO> --dataset glove-100-angular --dataset_size <SIZE>
```

**Arguments:**
- `--algorithm`: `milvus-hnsw`, `milvus-ivfflat`, `pgvector`, `pgvector_ivf`, `hnsw(faiss)`, `faiss-ivf`
- `--dataset`: Dataset name (e.g. `glove-100-angular`)
- `--dataset_size`: `small`, `medium`, or `large`
- `--force`: Re-run even if results exist
- `--parallelism`: Number of parallel workers (default: 1)

### Orchestrator: `starter.py`

`starter.py` runs multiple algorithms and dataset sizes and generates config via `make_yaml.py`:

```bash
python starter.py [--dataset_size small|medium|large]
```

Edit the `algo` list and `dataset_size` loop in `starter.py` to choose which algorithms and sizes to run.

### Single algorithm run

```bash
python run.py --algorithm milvus-hnsw --dataset glove-100-angular --dataset_size small
```

---

## Ablation study (Milvus segment size)

The ablation workflow tests how Milvus segment size (512 MB–16 GB) affects performance.

**Scripts (dedicated ablation workflow):**
- `starter_ablation.py`: Runs ablation across segment sizes and algorithms
- `run_ablation.py`: Entry point using `main_ablation.py`
- `make_yaml_ablation.py`: Generates config with segment-size-specific Docker tags
- `ann_benchmarks/main_ablation.py`, `runner_ablation.py`, `results_ablation.py`: Core logic with segment size support

**Run ablation:**

```bash
# Full ablation (all segment sizes, default algorithms)
python starter_ablation.py --dataset_size small

# Custom segment sizes
python starter_ablation.py --dataset_size medium --segment_sizes 1024 4096 16384

# Single segment size via run_ablation
python run_ablation.py --algorithm milvus-hnsw --dataset glove-100-angular --dataset_size small --segment_size 1024
```

**Results layout:** `results/ablation_seg{size}/MoRe_UPD_{size}_attidx_0/...`

---

## Plotting and analysis

Analysis scripts live in `analysis/`. Run them after generating benchmark results.

### 1. Build CSV from HDF5 results

```bash
cd analysis
python make_results.py
```

Useful options (check `--help`):
- Paths and dataset size
- Output CSV location

The output CSV is used by plotting scripts.

### 2. Plotting scripts

All plotting scripts expect CSVs under `results/` (or configurable paths).

| Script | Description |
|--------|-------------|
| `make_plots.py` | Basic throughput vs recall, recall vs selectivity |
| `make_plots_results1.py` | Throughput vs recall by selectivity (for paper) |
| `make_plots_results1_ivf.py` | Same, for IVF algorithms |
| `make_plots_results_ALL.py` | Combined plots for all dataset sizes |
| `make_plots_hnsw_vs_ivf_comparison.py` | HNSW vs IVF comparison |
| `make_plots_ablation_seg16gb.py` | Ablation: 1 GB vs 16 GB segment size |
| `make_plots_attidx_comparison.py` | Attribute index comparison |
| `make_plots_build_times.py` | Index build time plots |

**Example:**

```bash
cd analysis
python make_plots_results1.py
python make_plots_ablation_seg16gb.py  # After ablation runs
```

Some scripts hardcode paths (e.g. `ROOT_RESULTS`, `root_results`, `root_data`). Edit these at the top of each script if your repository path differs.

---

## Project layout

```
ann-benchmarks-HQ/
├── ann_benchmarks/
│   ├── algorithms/       # Algorithm implementations (faiss, faiss_hnsw, milvus, pgvector, pgvector_ivf)
│   ├── main.py           # Main benchmark entry
│   ├── main_ablation.py  # Ablation entry (segment size)
│   ├── runner.py         # Run logic
│   ├── runner_ablation.py
│   ├── results.py       # Result storage
│   ├── results_ablation.py  # Ablation result paths
│   ├── datasets.py      # Dataset loading (MoRe)
│   └── definitions.py
├── analysis/            # Plotting and analysis scripts
├── data/
│   └── datasets/        # MoRe dataset (MoRe_small, MoRe_medium, MoRe_large)
├── make_yaml.py         # Config generator for main runs
├── make_yaml_ablation.py
├── run.py               # Main entry: python run.py ...
├── run_ablation.py      # Ablation entry
├── starter.py           # Orchestrator for main runs
├── starter_ablation.py   # Orchestrator for ablation
└── requirements.txt
```

---

## Tested algorithms summary

| Backend | Algorithm | Config / notes |
|---------|-----------|----------------|
| **pgvector** | HNSW | `ann_benchmarks/algorithms/pgvector/config.yml` |
| **pgvector** | IVFFlat | `ann_benchmarks/algorithms/pgvector_ivf/config.yml` |
| **FAISS** | HNSW | `ann_benchmarks/algorithms/faiss_hnsw/config.yml` |
| **FAISS** | IVF | `ann_benchmarks/algorithms/faiss/config.yml` |
| **Milvus** | HNSW | `ann_benchmarks/algorithms/milvus/config.yml` |
| **Milvus** | IVFFlat | Same config |
| **Milvus** | IVFSQ8, IVFPQ, SCANN | Check config for enabled/disabled |

---

## License

See the original [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) license. Milvus components may have additional Apache-2.0 terms.
