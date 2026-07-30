# Task 1 — Hard / Superhard QO Eval Notes

## What was broken / fixed

- Runner only knew flex `fid` workloads; hard/superhard packs were on disk but not loadable.
- FAISS pre-filter hardcoded `X_attr >= threshold` (broke movies superhard `year` / `<=`).
- pgvector stored a single `filter_attr` column (broke multi-attr reviews / movies year).
- QO analysis assumed `.../fid{N}/...` flex layout.

## Workload modes

| Flag | Behavior |
|------|----------|
| `--workload flex` (default) | Unchanged: fid loop, k∈{10,20,40}, `results/MoRe_UPD_large_attidx_*/fid*/...` |
| `--workload hard` | Packs under `hard_queries/`, per-query filters, k=10, `results/MoRe_UPD_large_hard_{table}/10/...` |
| `--workload superhard` | Same for `superhard_queries/` |

Re-run flex (legacy): `python starter.py --dataset_size large`  
Re-run hard only: `python starter.py --dataset_size large --workload hard`  
Full campaign: `bash scripts/run_task1_hard_superhard_campaign.sh` (prefer nohup; logs in `logs/task1_hard_superhard/`).

## Analysis

```bash
python analysis/query_optimizer_hard_analysis.py --hardness hard --table movies
python analysis/query_optimizer_hard_analysis.py --hardness superhard --table reviews
```

Outputs: `analysis/plots/query_optimizer_{hard|superhard}_{movies|reviews}/` including tertile QPS–recall PNGs and `tertile_edges_*.csv`.

Estimated GLS (ρ̂): `scripts/compute_gls_estimates_hard.py` → `data/datasets/MoRe_large/stats/gls_est_{hardness}_{table}.csv`. If the `glscore` env is unavailable, exact-GLS and Post_Hardness tertiles still run; estimated-GLS tertiles are deferred.

Analysis recall uses cached **angular** filtered GT (`stats/angular_gt_{hardness}_{table}_k10.npy`) because pack GT is L2-based while the runner is angular.

## Campaign status

- Running under nohup (`scripts/run_task1_hard_superhard_campaign.sh`)
- Logs: `logs/task1_hard_superhard/master_*.log`; status: `logs/task1_hard_superhard/status_*.txt`
- Monitor: `tail -f logs/task1_hard_superhard/master_*.log`
- pgvector Dockerfile fixed: `apt-get update` before `tzdata` (stale mirror 404)
- Smoke confirmed: hard `faiss-flat` wrote 1000×10 results for movies + reviews

## Observations (fill after campaign)

- **Oracle plan-mix hard → superhard (movies):** _pending campaign_
- **Oracle plan-mix hard → superhard (reviews):** _pending campaign_
- **Movies vs reviews plan mix:** _pending campaign_
- **Hardness-bin / tertile frontier shifts:** _pending campaign_
- **ρ̂ tertiles:** _pending / deferred if estimator missing_

## Logs

- Master + per-stage: `ANN-benchmark-HQ/logs/task1_hard_superhard/`
- Status file: `logs/task1_hard_superhard/status_*.txt` (`status=complete` on success)
