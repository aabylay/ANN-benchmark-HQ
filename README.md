# ANN-Benchmarks Extension for Filtered Vector Search

This repository extends [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) for **Filtered ANNS** queries on the MoRe dataset. Benchmarks measure recall, query latency, and throughput under varying filter selectivity.

The current branch (`fannsqo`) focuses on **query plan selection**: comparing pre-filtering, post-filtering, and brute-force execution strategies across FAISS and pgvector, then analyzing which plan is best per query. Evaluation on the large MoRe dataset has been run; raw HDF5 results live under `results/`, and query-optimizer analysis outputs under `analysis/plots/query_optimizer/`.

## Overview

- **Custom dataset**: MoRe (Movies & Reviews) — movies and reviews with embeddings and filterable attributes.
- **Two systems**: FAISS and pgvector (Milvus support remains in the tree for legacy ablation runs but is no longer part of the active benchmark sweep).
- **Eight query plans** (index type × filter strategy):

| System | Plan | Algorithm name | Filter strategy |
|--------|------|----------------|-----------------|
| FAISS | Brute force | `faiss-flat` | Bitset pre-filter (`IDSelectorBitmap`) |
| FAISS | HNSW pre | `hnsw(faiss)` | Bitset pre-filter during graph search |
| FAISS | HNSW post | `hnsw(faiss)-post` | Over-fetch candidates, filter in NumPy |
| FAISS | IVF pre | `faiss-ivf` | Bitset pre-filter during IVF search |
| FAISS | IVF post | `faiss-ivf-post` | Over-fetch candidates, filter in NumPy |
| pgvector | Brute force | `pgvector_bf` | SQL `WHERE` post-filter, no index |
| pgvector | HNSW | `pgvector` | Iterative scan post-filter |
| pgvector | IVF | `pgvector_ivf` | Iterative scan post-filter |

**Post-filtering (FAISS)** is implemented in `ann_benchmarks/algorithms/faiss/postfilter.py`. For a target `k` and filter selectivity `sel`, the search pool size is `min(1000, ceil(k / sel * gamma))` (default `gamma=1.0`); candidates are then filtered in search order until `k` matches remain.

**Pre-filtering (FAISS)** uses `faiss.IDSelectorBitmap` built from the named attribute column and filter predicate (operators `>=`, `<=`, `>`, `<`, `=`, `!=`). pgvector stores multiple numeric FLOAT columns and emits `WHERE {attr} {op} {value}`.

Index construction parameters are fixed across the sweep: HNSW uses `M=16`, `efConstruction=128`; IVF uses `nlist ≈ sqrt(|D|)` (auto when `clusters=0`). Only search parameters are swept (`efSearch` for HNSW, `nprobe`/`probes` for IVF; current starter grid `efSearch ∈ {40,60,80}`).

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

The benchmark uses MoReVec datasets in HDF5 format under `data/datasets/`:

```
data/datasets/MoRe_{size}/
├── datasets/           # Train embeddings (movies, reviews)
│   ├── movies_dataset_0.hdf5
│   └── reviews_dataset_0.hdf5
├── filters/            # Flex filter bank + selectivities
│   ├── movies_filters_0.hdf5
│   └── reviews_filters_0.hdf5
├── queries/            # Flex workloads (one filter per file)
│   └── queries_flex_{type}_sim_0_{filter_id}.hdf5
├── hard_queries/       # Per-query hard packs (optional; generate locally)
│   ├── movies_hcbgen_match_pdf.hdf5
│   └── reviews_hcbgen_match_pdf.hdf5
├── superhard_queries/  # Per-query superhard packs (optional)
│   ├── movies_hcbgen_superhard.hdf5
│   └── reviews_hcbgen_superhard.hdf5
└── stats/              # Filter statistics (GLS correlation, etc.)
    └── filter_stats_0_k2048.csv
```

Supported sizes: `small`, `medium`, `large`. Flex queries ship with MoReVec; hard/superhard packs are generated on top of the large checkout (see `scripts/generate_hcbgen_hard_queries.py`, `scripts/generate_superhard_queries.py`, and pack READMEs under `hard_queries/` / `superhard_queries/`).

MoReVec datasets: [Google Drive folder](https://drive.google.com/drive/folders/1AqAVI8ASROqrFCQdEMPB8RNzPwilijRp?usp=drive_link)

Or run:

```bash
python load_morevec.py
```

### 3. Docker images

Build the images used by the current benchmark sweep:

```bash
docker build -t ann-benchmarks-faiss ann_benchmarks/algorithms/faiss/
docker build -t ann-benchmarks-pgvector ann_benchmarks/algorithms/pgvector/
```

The FAISS image installs `faiss-cpu==1.12.0`, which provides `SearchParametersHNSW`, `SearchParametersIVF`, and `IDSelectorBitmap` required for pre-filtering.

---

## Running benchmarks

**Arguments** (via `run.py` / Docker):
- `--algorithm`: any plan name from the table above
- `--dataset_size`: `small`, `medium`, or `large`
- `--workload`: `flex` (default), `hard`, or `superhard`

### Workload modes

| `--workload` | Queries | Filters | `k` | Result root |
|--------------|---------|---------|-----|-------------|
| `flex` (**default**) | `queries/queries_flex_*` | one filter per `fid` | `{10,20,40}` | `results/MoRe_UPD_{size}_attidx_{0\|1}/fid{N}/...` |
| `hard` | `hard_queries/{table}_hcbgen_match_pdf.hdf5` | **per-query** | `10` only | `results/MoRe_UPD_{size}_hard_{table}/10/...` |
| `superhard` | `superhard_queries/{table}_hcbgen_superhard.hdf5` | **per-query** | `10` only | `results/MoRe_UPD_{size}_superhard_{table}/10/...` |

No `--workload` flag ⇒ same as today's flex large-MoRe campaign.

### Orchestrator: `starter.py`

`starter.py` generates algorithm config via `make_yaml.py` and runs the benchmark sweep:

```bash
# Flex (default) — FAISS plans, fid loop, k∈{10,20,40}
python starter.py --dataset_size large

# Hard or superhard packs — all 8 FAISS/pgvector plans, k=10
python starter.py --dataset_size large --workload hard
python starter.py --dataset_size large --workload superhard

# Optional: restrict algorithms
python starter.py --dataset_size large --workload hard --algorithms faiss-flat,pgvector_bf
```

End-to-end hard+superhard campaign (Docker rebuild → BF smoke → 8-plan sweep → QO ×4), intended under `nohup`:

```bash
mkdir -p logs/task1_hard_superhard
nohup bash scripts/run_task1_hard_superhard_campaign.sh \
  > logs/task1_hard_superhard/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

See `NOTES_task1_hard_superhard.md` for status/log locations and acceptance notes.

### Config generation: `make_yaml.py`

`make_yaml.py` writes the per-algorithm `config.yml` with fixed construction params and search-parameter sweeps. It is called automatically by `starter.py`; you can also invoke it directly when running individual algorithms.

---

## Query optimizer analysis

### Flex workloads

`analysis/query_optimizer_analysis.py` is the plan-selection tool for the **flex** `fid` result tree. It:

1. Builds a per-query results CSV from HDF5 benchmark outputs.
2. Verifies brute-force plans (`faiss-flat`, `pgvector_bf`) achieve recall = 1.0.
3. For each ANN method, picks the hyperparameter that maximizes QPS subject to mean recall ≥ 0.95.
4. For each query, picks the best plan among tuned ANN methods and brute-force.
5. Produces scatter plots (selectivity vs GLS correlation, colored by best plan), QPS–recall frontiers, in-system oracle speedup charts, and CSV summaries.

**Run** (after flex benchmark results exist):

```bash
cd analysis
python query_optimizer_analysis.py \
  --results-dir results/MoRe_UPD_large_attidx_0 \
  --dataset-size large \
  --filter-stats data/datasets/MoRe_large/stats/filter_stats_0_k2048.csv \
  --output-dir plots/query_optimizer
```

**Outputs** (written to `analysis/plots/query_optimizer/` by default):

| File | Description |
|------|-------------|
| `all_query_results.csv` | Per-query recall, latency, QPS for every algorithm/hyperparam |
| `best_hyperparameters.csv` | Tuned search params per ANN method at recall ≥ 0.95 |
| `hyperparam_sweep.csv`, `hyperparam_recommendations.csv` | Full sweep and guidance for extending param ranges |
| `best_plan_per_query.csv` | Oracle best plan per query |
| `system_oracle_speedup.csv`, `system_oracle_speedup.png` | Speedup from picking HNSW vs IVF per query |
| `best_plan_scatter_*.png` | Selectivity vs GLS correlation, colored by winning plan |
| `best_plan_by_selectivity_bin.png` | Plan distribution across selectivity bins |
| `qps_recall_frontier_*.png` | QPS–recall curves with chosen hyperparameters marked |

### Hard / superhard workloads

`analysis/query_optimizer_hard_analysis.py` consumes
`results/MoRe_UPD_{size}_{hard|superhard}_{movies|reviews}/` and joins per-query
σ / Post_Hardness / exact GLS from the pack (plus optional ρ̂ CSV). It emits the
same oracle CSVs/plots style as flex, plus **tertile-binned QPS–recall** frontiers.

```bash
python analysis/query_optimizer_hard_analysis.py --hardness hard --table movies
python analysis/query_optimizer_hard_analysis.py --hardness superhard --table reviews
# → analysis/plots/query_optimizer_{hard|superhard}_{movies|reviews}/
```

Optional ρ̂ cache (same GLS-CorE settings as flex large):

```bash
conda activate glscore
python scripts/compute_gls_estimates_hard.py --hardness hard --tables movies reviews
```

Recall for hard packs uses cached **angular** filtered GT under
`data/datasets/MoRe_large/stats/angular_gt_{hardness}_{table}_k10.npy` (pack
neighbor lists were produced with L2; the runner uses angular).

### Per-query debugging: `run_query_analysis.py`

For detailed inspection of a single filter/workload on pgvector (returns IDs, recall, ground truth per query):

```bash
python run_query_analysis.py \
  --k 10 --filter_id 0 --ef_search 100 --nprobe 10 \
  --dataset_size large --dataset_type movies
```

---

## Legacy plotting and analysis

Older analysis scripts in `analysis/` operate on aggregated CSVs and remain available for paper/workshop figures:

| Script | Description |
|--------|-------------|
| `make_results.py` | Build CSV from HDF5 results |
| `make_plots_results_ALL.py` | Combined plots for all dataset sizes |
| `make_plots_hnsw_vs_ivf_comparison.py` | HNSW vs IVF comparison |
| `make_plots_build_times.py` | Index build time plots |
| `make_plots_4_vldb.py` | VLDB-style figure set |

Many scripts hardcode `ROOT_RESULTS` or similar paths at the top of the file — edit these if your checkout path differs.

---

## Legacy: Milvus ablation (optional)

Milvus algorithm code and ablation tooling are retained but not used in the current query-optimizer sweep:

- Docker: `ann-benchmarks-milvus-seg16384` (build via `ann_benchmarks/algorithms/milvus/`)
- Ablation scripts: `starter_ablation.py`, `run_ablation.py`, `make_yaml_ablation.py`
- Segment-size study: `./build_milvus_ablation.sh`

---

## Project layout

```
ANN-benchmark-HQ/
├── ann_benchmarks/
│   ├── algorithms/
│   │   ├── faiss/              # IVF pre/post, flat BF, postfilter.py
│   │   ├── faiss_hnsw/         # HNSW pre/post
│   │   ├── faiss_hnsw_post/    # config for hnsw(faiss)-post
│   │   ├── faiss_ivf_post/     # config for faiss-ivf-post
│   │   ├── faiss_flat/         # config for faiss-flat
│   │   ├── pgvector/           # HNSW (+ multi-attr SQL helpers)
│   │   ├── pgvector_ivf/       # IVF
│   │   ├── pgvector_bf/        # brute force
│   │   ├── pgvector_common.py  # shared multi-column table/WHERE helpers
│   │   └── milvus/             # legacy
│   ├── attrs.py                # attribute name → column maps
│   ├── main.py, runner.py, results.py
│   └── datasets.py             # MoRe flex + hard/superhard pack loading
├── analysis/
│   ├── query_optimizer_analysis.py        # flex QO
│   ├── query_optimizer_hard_analysis.py   # hard/superhard QO + tertile frontiers
│   └── make_plots_*.py
├── scripts/
│   ├── generate_hcbgen_hard_queries.py
│   ├── generate_superhard_queries.py
│   ├── compute_gls_estimates_hard.py
│   └── run_task1_hard_superhard_campaign.sh
├── data/datasets/              # MoRe_small, MoRe_medium, MoRe_large
├── results/                    # HDF5 benchmark outputs
├── NOTES_task1_hard_superhard.md
├── make_yaml.py
├── starter.py                  # --workload flex|hard|superhard
├── run.py
└── requirements.txt
```

---

## License

See the original [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) license.
