#!/usr/bin/env bash
# Task 1 campaign: rebuild → BF smoke → hard+superhard 8-plan sweep → QO ×4
#                  → optimal-plan scatters ×4 [→ learned QO prototype ×4]
#
# Usage (from ANN-benchmark-HQ):
#   nohup bash scripts/run_task1_hard_superhard_campaign.sh \
#     > logs/task1_hard_superhard/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Flags:
#   --with-learned-qo   also run analysis/qo_prototype.py per (hardness, table)
#                       (off by default to keep the campaign lighter)
set -euo pipefail

WITH_LEARNED_QO=0
for arg in "$@"; do
  case "$arg" in
    --with-learned-qo) WITH_LEARNED_QO=1 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

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
# Status file: first line is status=..., followed by one stage_done=... line
# per completed stage (so artifact verification can list e.g. optimal_scatter).
STAGE_LINES=""
set_status() { printf 'status=%s\n%s' "$1" "$STAGE_LINES" > "$STATUS_FILE"; }
mark_stage() { STAGE_LINES+="stage_done=$1"$'\n'; set_status "running"; }
fail() { set_status "failed"; log "FAIL: $*"; exit 1; }

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
mark_stage "pack_gate"

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
mark_stage "docker_rebuild"

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
mark_stage "bf_smoke"

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
mark_stage "sweep"

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
mark_stage "gls_estimates"

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
mark_stage "qo_analysis"

# ---------------------------------------------------------------------------
# 5b. Optimal-plan scatters ×4 (exact GLS always; estimated GLS when ρ̂ exists)
# ---------------------------------------------------------------------------
for w in hard superhard; do
  for t in movies reviews; do
    qo="analysis/plots/query_optimizer_${w}_${t}"
    csv="$qo/all_query_results.csv"
    if [[ ! -f "$csv" ]]; then
      log "Optimal scatter skipped: missing $csv"
      MISSING+=("$csv")
      continue
    fi
    # QO analysis materializes gls_est/all_query_results.csv (gls_correlation
    # := rho-hat) whenever the pack has estimates. Pass it when present; else
    # optimal_plan_scatter.py falls back to the gls_correlation_est column of
    # the exact CSV, and if neither exists it emits exact-GLS plots only.
    est_csv="$qo/gls_est/all_query_results.csv"
    EST_ARGS=()
    if [[ -f "$est_csv" ]]; then
      EST_ARGS=(--gls-est-results-csv "$est_csv")
      log "Optimal scatter $w/$t (exact + estimated GLS)"
    else
      log "Optimal scatter $w/$t (exact GLS; no rho-hat CSV at $est_csv)"
    fi
    if python analysis/optimal_plan_scatter.py \
         --results-csv "$csv" \
         "${EST_ARGS[@]}" \
         --k 10 \
         --eps 0.1 \
         --output-dir "$qo/optimal" \
         --gls-est-output-dir "$qo/optimal/gls_est" \
         2>&1 | tee "$LOG_DIR/optimal_${w}_${t}_${TS}.log"; then
      log "Optimal scatter ok: $qo/optimal"
    else
      MISSING+=("$qo/optimal")
      log "Optimal scatter failed: $qo/optimal"
    fi
  done
done
mark_stage "optimal_scatter"

# ---------------------------------------------------------------------------
# 5c. Optional learned QO prototype ×4 (--with-learned-qo)
# ---------------------------------------------------------------------------
if ((WITH_LEARNED_QO)); then
  for w in hard superhard; do
    for t in movies reviews; do
      qo="analysis/plots/query_optimizer_${w}_${t}"
      if [[ ! -f "$qo/all_query_results.csv" ]]; then
        log "Learned QO skipped: missing $qo/all_query_results.csv"
        continue
      fi
      log "Learned QO prototype $w/$t"
      if python analysis/qo_prototype.py \
           --results "$qo/all_query_results.csv" \
           --out "analysis/plots/qo_prototype_${w}_${t}" \
           2>&1 | tee "$LOG_DIR/qo_prototype_${w}_${t}_${TS}.log"; then
        log "Learned QO ok: analysis/plots/qo_prototype_${w}_${t}"
      else
        MISSING+=("analysis/plots/qo_prototype_${w}_${t}")
        log "Learned QO failed: $w/$t"
      fi
    done
  done
  mark_stage "learned_qo"
fi

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
# Optimal-scatter figures (per algorithm) under optimal/faiss_all/, plus the
# rates CSV. Estimated-GLS twins are required only when rho-hat was available.
OPTIMAL_SCATTER_BASES=(
  optimal_scatter_hnsw_faiss
  optimal_scatter_hnsw_faiss_post
  optimal_scatter_faiss_ivf
  optimal_scatter_faiss_ivf_post
  optimal_scatter_faiss_flat
)
for w in hard superhard; do
  for t in movies reviews; do
    out="analysis/plots/query_optimizer_${w}_${t}"
    for f in "${REQ_FILES[@]}"; do
      if [[ ! -f "$out/$f" ]]; then
        MISSING+=("$out/$f")
      fi
    done
    for b in "${OPTIMAL_SCATTER_BASES[@]}"; do
      if [[ ! -f "$out/optimal/faiss_all/${b}_${t}.png" ]]; then
        MISSING+=("$out/optimal/faiss_all/${b}_${t}.png")
      fi
    done
    if [[ ! -f "$out/optimal/faiss_all/optimal_plan_rates.csv" ]]; then
      MISSING+=("$out/optimal/faiss_all/optimal_plan_rates.csv")
    fi
    if [[ -f "$out/gls_est/all_query_results.csv" ]]; then
      for b in "${OPTIMAL_SCATTER_BASES[@]}"; do
        if [[ ! -f "$out/optimal/gls_est/faiss_all/${b}_${t}.png" ]]; then
          MISSING+=("$out/optimal/gls_est/faiss_all/${b}_${t}.png")
        fi
      done
      if [[ ! -f "$out/optimal/gls_est/faiss_all/optimal_plan_rates.csv" ]]; then
        MISSING+=("$out/optimal/gls_est/faiss_all/optimal_plan_rates.csv")
      fi
    fi
  done
done

if ((${#MISSING[@]})); then
  log "Missing artifacts:"
  printf '  %s\n' "${MISSING[@]}"
  set_status "failed"
  echo "missing:" >> "$STATUS_FILE"
  printf '  %s\n' "${MISSING[@]}" >> "$STATUS_FILE"
  exit 1
fi

mark_stage "verify_artifacts"
set_status "complete"
log "Campaign complete. Status written to $STATUS_FILE"
log "Results under results/MoRe_UPD_large_{hard,superhard}_{movies,reviews}/"
log "QO under analysis/plots/query_optimizer_{hard,superhard}_{movies,reviews}/"
log "Optimal scatters under analysis/plots/query_optimizer_*/optimal/{,gls_est/}"
