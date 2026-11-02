# ANN-Benchmarks Extension for Filtered Vector Search

This repository extends [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) for **Filtered ANNS** queries on the MoRe dataset. Benchmarks measure recall, query latency, and throughput under varying filter selectivity.

The current branch (`fannsqo`) focuses on **query plan selection**: comparing pre-filtering, post-filtering, and brute-force execution strategies across FAISS and pgvector, then analyzing which plan is best per query. The two corpora are **MoReVec** (flex, hard, and superhard workloads) and **CaseLaw** (native filtered workload, 7,414,023 vectors). Raw HDF5 results live under `results/`, and query-optimizer analysis outputs under `analysis/plots/`.

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

Index construction parameters are fixed across the sweep: HNSW uses `M=16`, `efConstruction=128`; IVF uses `nlist ≈ sqrt(|D|)` (auto when `clusters=0`). Only search parameters are swept (`efSearch` for HNSW, `nprobe`/`probes` for IVF; current starter grid `efSearch ∈ {10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500, 600, 800}` — 17 points, comparable in density to the 18-point large IVF probe grid).

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
# MoReVec flex — FAISS plans, fid loop, k∈{10,20,40}
python starter.py --dataset morevec --dataset_size large

# MoReVec hard or superhard packs — all 8 FAISS/pgvector plans, k=10
python starter.py --dataset morevec --dataset_size large --workload hard
python starter.py --dataset morevec --dataset_size large --workload superhard

# CaseLaw native workload (official filtered queries; not an HCBGen pack)
python starter.py --dataset caselaw

# Both corpora. This is also the default when --dataset is omitted.
python starter.py --dataset both --dataset_size large

# Optional: restrict algorithms
python starter.py --dataset morevec --dataset_size large --workload hard --algorithms faiss-flat,pgvector_bf
```

End-to-end hard+superhard campaign (Docker rebuild → BF smoke → 8-plan sweep → QO ×4 → optimal-plan scatters ×4; add `--with-learned-qo` for the qo_prototype stage), intended under `nohup`:

```bash
mkdir -p logs/task1_hard_superhard
nohup bash scripts/run_task1_hard_superhard_campaign.sh \
  > logs/task1_hard_superhard/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The campaign log is the file named in that command. `--dataset` defaults to `both`, so a MoRe-only campaign must pass `--dataset morevec`.

### Config generation: `make_yaml.py`

`make_yaml.py` writes the per-algorithm `config.yml` with fixed construction params and search-parameter sweeps. It is called automatically by `starter.py`; you can also invoke it directly when running individual algorithms.

---

## Query optimizer analysis

### Flex workloads

`analysis/query_optimizer_analysis.py` is the plan-selection tool for the **flex** `fid` result tree. It:

1. Builds a per-query results CSV from HDF5 benchmark outputs.
2. Verifies brute-force plans (`faiss-flat`, `pgvector_bf`) achieve recall = 1.0.
3. For each ANN method, picks the hyperparameter that maximizes QPS subject to mean recall ≥ the target (default **0.95**; also **0.80**, **0.90**, and **0.99**).
4. For each query, picks the best plan among tuned ANN methods and brute-force.
5. Produces scatter plots (selectivity vs GLS correlation, colored by best plan), QPS–recall frontiers, in-system oracle speedup charts, and CSV summaries.

Shared CSVs stay at the output-dir root. Threshold-dependent plots and CSVs nest under `r080/` / `r090/` / `r095/` / `r099/` (`recall_tag(t) = f"r{round(t*100):03d}"`).

**Run** (after flex benchmark results exist):

```bash
cd analysis
python query_optimizer_analysis.py \
  --results-dir results/MoRe_UPD_large_attidx_0 \
  --dataset-size large \
  --filter-stats data/datasets/MoRe_large/stats/filter_stats_0_k2048.csv \
  --output-dir plots/query_optimizer \
  --recall-target 0.95
# also: --recall-target 0.80  → plots/query_optimizer/r080/
#       --recall-target 0.90  → plots/query_optimizer/r090/
#       --recall-target 0.99  → plots/query_optimizer/r099/
```

**Outputs** (shared CSVs at `analysis/plots/query_optimizer/`; threshold-dependent files under `rXXX/`):

| File | Description |
|------|-------------|
| `all_query_results.csv` | Per-query recall, latency, QPS for every algorithm/hyperparam (workload root) |
| `rXXX/best_hyperparameters.csv` | Tuned search params per ANN method at the recall target |
| `hyperparam_sweep.csv` | Full HP × mean-recall/QPS table (workload root) |
| `rXXX/hyperparam_recommendations.csv` | Guidance for extending param ranges at the target |
| `rXXX/best_plan_per_query.csv` | Oracle best plan per query |
| `rXXX/system_oracle_speedup.csv`, `rXXX/system_oracle_speedup.png` | Speedup from picking HNSW vs IVF per query |
| `rXXX/best_plan_scatter_*.png` | Selectivity vs GLS correlation, colored by winning plan |
| `rXXX/best_plan_by_selectivity_bin.png` | Plan distribution across selectivity bins |
| `rXXX/qps_recall_frontier_*.png` | QPS–recall curves with chosen hyperparameters marked |

### Hard / superhard workloads

`analysis/query_optimizer_hard_analysis.py` consumes
`results/MoRe_UPD_{size}_{hard|superhard}_{movies|reviews}/` and joins per-query
σ / Post_Hardness / exact GLS from the pack (plus optional ρ̂ CSV). It emits the
same oracle CSVs/plots style as flex, plus **per-algorithm QPS–recall frontiers**
by Post_Hardness (red) and estimated GLS (green) tertiles.

```bash
python analysis/query_optimizer_hard_analysis.py --hardness hard --table movies
python analysis/query_optimizer_hard_analysis.py --hardness superhard --table reviews --recall-target 0.90
# → analysis/plots/query_optimizer_{hard|superhard}_{movies|reviews}/
#    shared: all_query_results.csv, runtime/, qps_recall_tertiles_per_algo/
#    per target: r080/ r090/ r095/ r099/ (default --recall-target 0.95 → r095/)
```

Optional ρ̂ cache (same GLS-CorE settings as flex large):

```bash
conda activate glscore
python scripts/compute_gls_estimates_hard.py --hardness hard --tables movies reviews
```

**Optimal-plan scatters.** After the QO analysis, the campaign runs
`analysis/optimal_plan_scatter.py` for each `(hardness, table)` × `{0.80, 0.90, 0.95, 0.99}`:
per query, the best latency `t_b` is the fastest plan with recall ≥ the target, and
an (algo, HP) is near-optimal when `recall ≥ target` and `runtime ≤ t_b·(1+eps)`
(default `eps = 0.1`). One figure per algorithm, one subplot per search HP, axes =
selectivity × GLS. Exact-GLS figures go to `…/rXXX/optimal/`; when ρ̂ is available
(the `gls_est/all_query_results.csv` twin or the `gls_correlation_est` column)
a parallel tree is written to `…/rXXX/optimal/gls_est/`, otherwise the est pass is
skipped with a log line. Hard/superhard packs are k=10 only, so R≥0.80 is 8/10+,
R≥0.90 is 9/10+, and R≥0.95 and R≥0.99 both require 10/10 hits (figures include
that footnote). Manual invocation:

```bash
python analysis/optimal_plan_scatter.py \
  --results-csv analysis/plots/query_optimizer_hard_movies/all_query_results.csv \
  --gls-est-results-csv analysis/plots/query_optimizer_hard_movies/gls_est/all_query_results.csv \
  --k 10 --eps 0.1 --recall-target 0.95 \
  --output-dir analysis/plots/query_optimizer_hard_movies/r095/optimal \
  --gls-est-output-dir analysis/plots/query_optimizer_hard_movies/r095/optimal/gls_est
```

**Learned QO prototype** (`analysis/qo_prototype.py`, campaign flag
`--with-learned-qo`) evaluates the per-query plan+HP predictor on the same
CSVs (always read from the workload-root `all_query_results.csv` and sibling
`gls_est/`) → `analysis/plots/qo_prototype_{hardness}_{table}/rXXX/`. Per-plan HP grids
are derived from the swept hyperparams in the results CSV (the hardcoded
flex-era grids are only a fallback). The `always-HNSW-pre (tuned)` /
`always-IVF-pre (tuned)` baselines run each plan per query at the lowest swept
HP reaching the recall target (default 0.95; also 0.80, 0.90, and 0.99; max swept HP if none does). The oracle point is,
per query, the fastest plan among those reaching the recall target, each ANN
plan at its cheapest swept HP achieving it, with BF always a candidate
(recall 1.0). Note the evaluation does **not** charge optimizer decision
(model-inference) time — realised latency is only the measured runtime of the
chosen plan — so in the full-BF limit the gamma-sweep endpoint coincides
exactly with the always-BF point.

```bash
python analysis/qo_prototype.py \
  --results analysis/plots/query_optimizer_hard_movies/all_query_results.csv \
  --out analysis/plots/qo_prototype_hard_movies \
  --recall-target 0.90
# → analysis/plots/qo_prototype_hard_movies/r090/
# also: --recall-target 0.80 → analysis/plots/qo_prototype_hard_movies/r080/
```

`rXXX/qps_recall_plans_{exact,estimated}.png` is the throughput-vs-recall figure:
static HNSW-pre and IVF-pre curves (one global `efSearch` or `nprobe` for every
query), plus single points for brute force (recall 1.0), the per-query oracle,
the full query optimizer, and physical-only optimizers. A physical-only
optimizer keeps the logical plan fixed (`HNSW-pre` or `IVF-pre` / IVFFlat) and
only predicts `efSearch` or `nprobe`, using the same cross-validated model as
the full optimizer.

The cached-data GLS contribution study covers the original flex
movies/reviews/pooled workloads plus the six hard/superhard workloads at all
four recall targets:

```bash
GLS_CONTRIBUTION_STUDY=1 bash scripts/run_qo_prototype_hard_superhard.sh
# → analysis/plots/gls_contribution_summary.png (9 workloads × 4 targets)
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
├── datasets_etl/               # CaseLaw pack builder
├── make_yaml.py
├── starter.py                  # --workload flex|hard|superhard
├── run.py
└── requirements.txt
```

---

## License

See the original [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) license.
