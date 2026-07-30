#!/usr/bin/env bash
# Task 1 campaign: rebuild → BF smoke → hard+superhard 8-plan sweep → QO ×4
#
# Usage (from ANN-benchmark-HQ):
#   nohup bash scripts/run_task1_hard_superhard_campaign.sh \
#     > logs/task1_hard_superhard/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/task1_hard_superhard"
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/status_${TS}.txt"
PID_FILE="$LOG_DIR/campaign.pid"
MASTER_LOG="$LOG_DIR/master_${TS}.log"

echo "$$" > "$PID_FILE"
exec > >(tee -a "$MASTER_LOG") 2>&1

log() { echo "[$(date -Is)] $*"; }
fail() { echo "status=failed" > "$STATUS_FILE"; log "FAIL: $*"; exit 1; }
set_status() { echo "status=$1" > "$STATUS_FILE"; }

set_status "running"
log "Task 1 hard/superhard campaign starting (pid=$$)"
log "ROOT=$ROOT"

# ---------------------------------------------------------------------------
# 0. Pack presence gate
# ---------------------------------------------------------------------------
for p in \
  data/datasets/MoRe_large/hard_queries/movies_hcbgen_match_pdf.hdf5 \
  data/datasets/MoRe_large/hard_queries/reviews_hcbgen_match_pdf.hdf5 \
  data/datasets/MoRe_large/superhard_queries/movies_hcbgen_superhard.hdf5 \
  data/datasets/MoRe_large/superhard_queries/reviews_hcbgen_superhard.hdf5
do
  [[ -f "$p" ]] || fail "missing pack $p"
done
log "All four packs present"

# ---------------------------------------------------------------------------
# 1. Rebuild Docker images (faiss required; pgvector may reuse cache after apt fix)
# ---------------------------------------------------------------------------
log "Rebuilding ann-benchmarks-faiss and ann-benchmarks-pgvector"
# Avoid full base rebuild every time if base already exists — still rebuild algos.
if ! docker image inspect ann-benchmarks >/dev/null 2>&1; then
  docker build --rm -t ann-benchmarks -f ann_benchmarks/algorithms/base/Dockerfile . \
    2>&1 | tee "$LOG_DIR/docker_base_${TS}.log" || fail "base docker build"
fi
python install.py --algorithm faiss 2>&1 | tee "$LOG_DIR/docker_faiss_${TS}.log" || fail "faiss docker build"
python install.py --algorithm pgvector 2>&1 | tee "$LOG_DIR/docker_pgvector_${TS}.log" || fail "pgvector docker build"
log "Docker rebuild done"

# ---------------------------------------------------------------------------
# 2. BF smoke on all four packs (faiss-flat + pgvector_bf × hard + superhard)
# ---------------------------------------------------------------------------
log "BF smoke starting"
SMOKE_ALGOS="faiss-flat,pgvector_bf"
for w in hard superhard; do
  log "Smoke workload=$w algos=$SMOKE_ALGOS"
  python starter.py --dataset_size large --workload "$w" --algorithms "$SMOKE_ALGOS" \
    2>&1 | tee "$LOG_DIR/smoke_${w}_${TS}.log" || fail "smoke $w"
done
log "BF smoke finished"

# Quick recall sanity via analysis (BF-only dirs may be incomplete for full QO;
# we just ensure result files exist)
for w in hard superhard; do
  for t in movies reviews; do
    d="results/MoRe_UPD_large_${w}_${t}/10"
    [[ -d "$d/faiss-flat" ]] || fail "missing smoke results $d/faiss-flat"
    [[ -d "$d/pgvector_bf" ]] || fail "missing smoke results $d/pgvector_bf"
  done
done
log "BF smoke artifacts present"

# ---------------------------------------------------------------------------
# 3. Full 8-plan sweep (hard then superhard)
# ---------------------------------------------------------------------------
FULL_ALGOS="faiss-flat,hnsw(faiss),hnsw(faiss)-post,faiss-ivf,faiss-ivf-post,pgvector_bf,pgvector,pgvector_ivf"
for w in hard superhard; do
  log "Sweep workload=$w"
  python starter.py --dataset_size large --workload "$w" --algorithms "$FULL_ALGOS" \
    2>&1 | tee "$LOG_DIR/sweep_${w}_${TS}.log" || fail "sweep $w"
done
log "Sweep finished"

# ---------------------------------------------------------------------------
# 4. Optional GLS-CorE ρ̂ cache (best-effort; defer tertile-est if unavailable)
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  set +e
  for w in hard superhard; do
    log "Computing GLS estimates for workload=$w"
    conda run -n glscore python scripts/compute_gls_estimates_hard.py \
      --hardness "$w" --tables movies reviews \
      2>&1 | tee "$LOG_DIR/gls_est_${w}_${TS}.log"
  done
  set -e
else
  log "conda not found; skipping GLS-CorE estimates"
fi

# ---------------------------------------------------------------------------
# 5. QO analysis ×4
# ---------------------------------------------------------------------------
MISSING=()
for w in hard superhard; do
  for t in movies reviews; do
    log "QO analysis $w/$t"
    out="analysis/plots/query_optimizer_${w}_${t}"
    if python analysis/query_optimizer_hard_analysis.py \
         --hardness "$w" --table "$t" --output-dir "$out" \
         2>&1 | tee "$LOG_DIR/qo_${w}_${t}_${TS}.log"; then
      log "QO ok: $out"
    else
      MISSING+=("$out")
      log "QO failed: $out"
    fi
  done
done

# ---------------------------------------------------------------------------
# 6. Verify required artifacts
# ---------------------------------------------------------------------------
REQ_FILES=(
  all_query_results.csv
  best_hyperparameters.csv
  best_plan_per_query.csv
  qps_recall_by_post_hardness_tertiles.png
  qps_recall_by_gls_exact_tertiles.png
)
for w in hard superhard; do
  for t in movies reviews; do
    out="analysis/plots/query_optimizer_${w}_${t}"
    for f in "${REQ_FILES[@]}"; do
      if [[ ! -f "$out/$f" ]]; then
        MISSING+=("$out/$f")
      fi
    done
  done
done

if ((${#MISSING[@]})); then
  log "Missing artifacts:"
  printf '  %s\n' "${MISSING[@]}"
  echo "status=failed" > "$STATUS_FILE"
  echo "missing:" >> "$STATUS_FILE"
  printf '  %s\n' "${MISSING[@]}" >> "$STATUS_FILE"
  exit 1
fi

set_status "complete"
log "Campaign complete. Status written to $STATUS_FILE"
log "Results under results/MoRe_UPD_large_{hard,superhard}_{movies,reviews}/"
log "QO under analysis/plots/query_optimizer_{hard,superhard}_{movies,reviews}/"
