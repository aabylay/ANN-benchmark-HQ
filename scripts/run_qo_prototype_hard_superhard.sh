#!/usr/bin/env bash
# Full experimental pipeline: hard / superhard MoReVec workloads -> QO prototype.
#
# Stages (each idempotent; expensive ones are opt-in via env vars):
#   0. [RUN_SWEEP=1]    benchmark sweep campaign (docker, hours) — only needed
#                       when results/MoRe_UPD_large_{hard,superhard}_{table}/ are absent
#   1. [RUN_GLS_EST=1]  GLS-CorE rho-hat cache (glscore conda env) — only needed
#                       when data/datasets/MoRe_large/stats/gls_est_*.csv are absent
#   2.                  QO analysis x4 -> all_query_results.csv + gls_est/all_query_results.csv
#   2b.                 fixed-HP runtime + recall panels (dist + σ / Post_Hardness / ρ)
#   3.                  pooled per-hardness CSVs (movies + reviews), exact + estimated GLS
#   3a. [GLS_CONTRIBUTION_STUDY=1] recover and validate original flex caches
#   4.                  qo_prototype.py on 4 per-table + 2 pooled workloads
#                       [GLS_CONTRIBUTION_STUDY=1] adds 3 flex workloads and
#                       the three-way sigma/rho ablation
#   5.                  aggregate matched-recall GLS contribution summaries
#
# Usage (from ANN-benchmark-HQ):
#   bash scripts/run_qo_prototype_hard_superhard.sh
#   RUN_SWEEP=1 RUN_GLS_EST=1 bash scripts/run_qo_prototype_hard_superhard.sh   # from scratch
#   GLS_CONTRIBUTION_STUDY=1 bash scripts/run_qo_prototype_hard_superhard.sh    # cached-data ablation
#   RECALL_TARGETS="0.80 0.90 0.95 0.99"  # override recall targets (campaign only accepts these)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$HOME/miniconda3/envs/ann-hq/bin/python}"
log() { echo "[$(date -Is)] $*"; }

TARGETS="${RECALL_TARGETS:-0.80 0.90 0.95 0.99}"
GLS_CONTRIBUTION_STUDY="${GLS_CONTRIBUTION_STUDY:-0}"
case "$GLS_CONTRIBUTION_STUDY" in
  0) QO_EXTRA_ARGS=(); QO_LOG_NAME="run.log" ;;
  1) QO_EXTRA_ARGS=(--gls-contribution-study); QO_LOG_NAME="gls_contribution.log" ;;
  *) echo "GLS_CONTRIBUTION_STUDY must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$GLS_CONTRIBUTION_STUDY" == "1" && "${RUN_SWEEP:-0}" == "1" ]]; then
  echo "GLS_CONTRIBUTION_STUDY=1 is cached-only and cannot be combined with RUN_SWEEP=1" >&2
  exit 2
fi
recall_tag() {
  "$PY" -c 'import sys; print(f"r{round(float(sys.argv[1]) * 100):03d}")' "$1"
}
for rt in $TARGETS; do
  case "$rt" in
    0.8|0.80|0.9|0.90|0.95|0.99) ;;
    *) echo "Unsupported recall target '$rt' (allowed: 0.80 0.90 0.95 0.99)" >&2; exit 2 ;;
  esac
done
log "RECALL_TARGETS=$TARGETS"
log "GLS_CONTRIBUTION_STUDY=$GLS_CONTRIBUTION_STUDY"

# ---------------------------------------------------------------------------
# 0. Benchmark sweep campaign (opt-in; produces results/MoRe_UPD_large_*)
# ---------------------------------------------------------------------------
if [[ "${RUN_SWEEP:-0}" == "1" ]]; then
  log "Stage 0: running full sweep campaign"
  bash scripts/run_task1_hard_superhard_campaign.sh
else
  log "Stage 0: skipped (set RUN_SWEEP=1 to re-run the benchmark sweep)"
fi

# ---------------------------------------------------------------------------
# 1. Estimated GLS (rho-hat) cache (opt-in; needs glscore conda env)
# ---------------------------------------------------------------------------
if [[ "${RUN_GLS_EST:-0}" == "1" ]]; then
  log "Stage 1: tiered GLS estimates (pq_synopsis) for flex, hard, superhard"
  conda run -n glscore python scripts/compute_tiered_gls.py --partitioner pq_synopsis --tables movies reviews
else
  log "Stage 1: skipped (set RUN_GLS_EST=1 to recompute tiered rho-hat caches)"
fi
first_rt="${TARGETS%% *}"

# ---------------------------------------------------------------------------
# 2. QO analysis x4 -> all_query_results.csv (+ gls_est/ sibling for prototype)
#    CSV + tertiles once; threshold-dependent plots under r080/ r090/ r095/ r099/
# ---------------------------------------------------------------------------
for w in hard superhard; do
  for t in movies reviews; do
    for rt in $TARGETS; do
      log "Stage 2: QO analysis $w/$t recall=$rt"
      rebuild_args=()
      if [[ "$rt" == "$first_rt" && ( "${RUN_GLS_EST:-0}" == "1" || "${REBUILD_QO_CSV:-0}" == "1" ) ]]; then
        rebuild_args=(--rebuild-csv)
      fi
      "$PY" analysis/query_optimizer_hard_analysis.py \
        --hardness "$w" --table "$t" --recall-target "$rt" \
        "${rebuild_args[@]}"
    done
  done
done

# ---------------------------------------------------------------------------
# 2b. Fixed-HP runtime diagnostics (dist + σ / Post_Hardness / ρ scatters)
# ---------------------------------------------------------------------------
for w in hard superhard; do
  for t in movies reviews; do
    qo="analysis/plots/query_optimizer_${w}_${t}"
    log "Stage 2b: runtime plots $w/$t"
    "$PY" analysis/query_runtime_plots.py \
      --results-csv "$qo/all_query_results.csv" \
      --output-dir "$qo/runtime" \
      --k 10
  done
done

# ---------------------------------------------------------------------------
# 3. Pooled per-hardness CSVs (movies + reviews), matching the flex-run design
#    where Stage 1/2 models pool across tables and Stage 3 sees log(N)
# ---------------------------------------------------------------------------
for w in hard superhard; do
  log "Stage 3: pooling movies+reviews for $w"
  "$PY" - "$w" <<'EOF'
import sys
from pathlib import Path
import pandas as pd

w = sys.argv[1]
base = Path("analysis/plots")
out = base / f"query_optimizer_{w}_pooled"
(out / "gls_est").mkdir(parents=True, exist_ok=True)
for sub in ["", "gls_est"]:
    parts = [base / f"query_optimizer_{w}_{t}" / sub / "all_query_results.csv"
             for t in ["movies", "reviews"]]
    missing = [p for p in parts if not p.is_file()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    dest = out / sub / "all_query_results.csv"
    df.to_csv(dest, index=False)
    print(f"pooled {w} {sub or 'exact'}: {len(df)} rows -> {dest}")
EOF
done

if [[ "$GLS_CONTRIBUTION_STUDY" == "1" ]]; then
  log "Stage 3a: preparing original flex workload caches"
  if ! "$PY" - <<'EOF'
from pathlib import Path
import pandas as pd

exact = Path("analysis/plots/query_optimizer/all_query_results.csv")
estimated = exact.parent / "gls_est" / "all_query_results.csv"
if not exact.is_file() or not estimated.is_file():
    raise SystemExit(1)
for path in (exact, estimated):
    if "ef_construction" not in pd.read_csv(path, nrows=1).columns:
        raise SystemExit(1)
EOF
  then
    "$PY" analysis/query_optimizer_analysis.py \
      --results-dir results/MoRe_UPD_large_attidx_0 \
      --dataset-size large \
      --recall-target 0.95 \
      --rebuild-csv
  elif [[ "${RUN_GLS_EST:-0}" == "1" || "${REBUILD_QO_CSV:-0}" == "1" ]]; then
    log "Stage 3a: rebuilding flex caches with tiered GLS columns"
    "$PY" analysis/query_optimizer_analysis.py \
      --results-dir results/MoRe_UPD_large_attidx_0 \
      --dataset-size large \
      --recall-target 0.95 \
      --rebuild-csv
  fi

  log "Stage 3b: testing optimizer and preflighting all study inputs"
  PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider analysis/test_qo_prototype.py -q
  "$PY" - <<'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("analysis").resolve()))
from qo_prototype import load_sweep, validate_study_pair

flex = Path("analysis/plots/query_optimizer/all_query_results.csv")
specs = [
    ("flex_movies", flex, "movies"),
    ("flex_reviews", flex, "reviews"),
    ("flex_pooled", flex, None),
]
for workload in [
    "hard_movies",
    "hard_reviews",
    "superhard_movies",
    "superhard_reviews",
    "hard_pooled",
    "superhard_pooled",
]:
    specs.append((
        workload,
        Path("analysis/plots") / f"query_optimizer_{workload}" / "all_query_results.csv",
        None,
    ))
missing = []
paths = {}
for workload, exact, query_type in specs:
    estimated = exact.parent / "gls_est" / "all_query_results.csv"
    paths[workload] = (exact, estimated, query_type)
    missing.extend(str(path) for path in (exact, estimated) if not path.is_file())
if missing:
    raise SystemExit(f"missing GLS contribution inputs: {missing}")

for workload, (exact_path, estimated_path, query_type) in paths.items():
    exact = load_sweep(
        exact_path, require_rho=True, strict=True, query_type=query_type
    )
    estimated = load_sweep(
        estimated_path, require_rho=True, strict=True, query_type=query_type
    )
    validate_study_pair(exact, estimated)
    print(f"validated {workload}: {len(exact)} measured rows")
EOF
fi

# ---------------------------------------------------------------------------
# 4. QO prototype: 4 per-table workloads + 2 pooled per-hardness workloads
#    plus 3 original flex workloads in GLS-contribution mode, × recall targets.
# ---------------------------------------------------------------------------
WORKLOADS=(hard_movies hard_reviews superhard_movies superhard_reviews hard_pooled superhard_pooled)
if [[ "$GLS_CONTRIBUTION_STUDY" == "1" ]]; then
  WORKLOADS=(flex_movies flex_reviews flex_pooled "${WORKLOADS[@]}")
fi
for w in "${WORKLOADS[@]}"; do
  query_type_args=()
  results_csv="analysis/plots/query_optimizer_${w}/all_query_results.csv"
  case "$w" in
    flex_movies)
      results_csv="analysis/plots/query_optimizer/all_query_results.csv"
      query_type_args=(--query-type movies)
      ;;
    flex_reviews)
      results_csv="analysis/plots/query_optimizer/all_query_results.csv"
      query_type_args=(--query-type reviews)
      ;;
    flex_pooled)
      results_csv="analysis/plots/query_optimizer/all_query_results.csv"
      ;;
  esac
  for rt in $TARGETS; do
    tag="$(recall_tag "$rt")"
    out="analysis/plots/qo_prototype_${w}"
    mkdir -p "$out/$tag"
    log "Stage 4: qo_prototype on $w recall=$rt"
    "$PY" analysis/qo_prototype.py \
      --results "$results_csv" \
      --out "$out" \
      --recall-target "$rt" \
      "${query_type_args[@]}" \
      "${QO_EXTRA_ARGS[@]}" \
      2>&1 | tee "$out/$tag/$QO_LOG_NAME"
  done
done

if [[ "$GLS_CONTRIBUTION_STUDY" == "1" ]]; then
  log "Stage 5: aggregating GLS contribution summaries"
  "$PY" - "$TARGETS" <<'EOF'
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("analysis").resolve()))
from qo_prototype import plot_gls_contribution_summary

targets = sys.argv[1].split()
workloads = [
    "flex_movies",
    "flex_reviews",
    "flex_pooled",
    "hard_movies",
    "hard_reviews",
    "superhard_movies",
    "superhard_reviews",
    "hard_pooled",
    "superhard_pooled",
]

all_matched = []
summary = []
missing = []
for workload in workloads:
    for target_text in targets:
        target = float(target_text)
        tag = f"r{round(target * 100):03d}"
        root = Path("analysis/plots") / f"qo_prototype_{workload}" / tag
        matched_path = root / "gls_contribution_matched_recall.csv"
        summary_path = root / "gls_contribution_summary.csv"
        if not matched_path.is_file() or not summary_path.is_file():
            missing.extend(str(path) for path in (matched_path, summary_path) if not path.is_file())
            continue

        matched = pd.read_csv(matched_path)
        matched.insert(0, "recall_target", target)
        matched.insert(0, "workload", workload)
        all_matched.append(matched)

        point = pd.read_csv(summary_path)
        point.insert(0, "workload", workload)
        summary.append(point)

if missing:
    raise SystemExit(f"missing GLS contribution outputs: {missing}")

out = Path("analysis/plots")
all_matched = pd.concat(all_matched, ignore_index=True)
summary = pd.concat(summary, ignore_index=True)
all_matched.to_csv(out / "gls_contribution_matched_recall_all.csv", index=False)
summary.to_csv(out / "gls_contribution_summary.csv", index=False)
plot_gls_contribution_summary(summary, out / "gls_contribution_summary.png")
print(f"Saved aggregate GLS contribution results under {out}")
EOF
fi

log "Pipeline complete. Prototype outputs: analysis/plots/qo_prototype_{flex,hard,superhard}_{movies,reviews,pooled}/{r080,r090,r095,r099}/"
