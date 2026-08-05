#!/usr/bin/env bash
# Full experimental pipeline: hard / superhard MoReVec workloads -> QO prototype.
#
# Stages (each idempotent; expensive ones are opt-in via env vars):
#   0. [RUN_SWEEP=1]    benchmark sweep campaign (docker, hours) — only needed
#                       when results/MoRe_UPD_large_{hard,superhard}_{table}/ are absent
#   1. [RUN_GLS_EST=1]  GLS-CorE rho-hat cache (glscore conda env) — only needed
#                       when data/datasets/MoRe_large/stats/gls_est_*.csv are absent
#   2.                  QO analysis x4 -> all_query_results.csv + gls_est/all_query_results.csv
#   3.                  pooled per-hardness CSVs (movies + reviews), exact + estimated GLS
#   4.                  qo_prototype.py on 4 per-table + 2 pooled workloads
#
# Usage (from ANN-benchmark-HQ):
#   bash scripts/run_qo_prototype_hard_superhard.sh
#   RUN_SWEEP=1 RUN_GLS_EST=1 bash scripts/run_qo_prototype_hard_superhard.sh   # from scratch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$HOME/miniconda3/envs/ann-hq/bin/python}"
log() { echo "[$(date -Is)] $*"; }

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
  for w in hard superhard; do
    log "Stage 1: GLS estimates for workload=$w"
    conda run -n glscore python scripts/compute_gls_estimates_hard.py \
      --hardness "$w" --tables movies reviews
  done
else
  log "Stage 1: skipped (set RUN_GLS_EST=1 to recompute rho-hat caches)"
fi

# ---------------------------------------------------------------------------
# 2. QO analysis x4 -> all_query_results.csv (+ gls_est/ sibling for prototype)
# ---------------------------------------------------------------------------
for w in hard superhard; do
  for t in movies reviews; do
    log "Stage 2: QO analysis $w/$t"
    "$PY" analysis/query_optimizer_hard_analysis.py --hardness "$w" --table "$t"
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

# ---------------------------------------------------------------------------
# 4. QO prototype: 4 per-table workloads + 2 pooled per-hardness workloads
# ---------------------------------------------------------------------------
for w in hard_movies hard_reviews superhard_movies superhard_reviews hard_pooled superhard_pooled; do
  log "Stage 4: qo_prototype on $w"
  "$PY" analysis/qo_prototype.py \
    --results "analysis/plots/query_optimizer_${w}/all_query_results.csv" \
    --out "analysis/plots/qo_prototype_${w}" \
    2>&1 | tee "analysis/plots/qo_prototype_${w}.log"
done

log "Pipeline complete. Prototype outputs: analysis/plots/qo_prototype_{hard,superhard}_{movies,reviews,pooled}/"
