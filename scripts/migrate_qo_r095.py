#!/usr/bin/env python3
"""Idempotent one-shot: nest existing 0.95 QO / prototype artifacts under r095/.

Shared CSVs, runtime/, and qps_recall_tertiles_per_algo/ stay at the workload root.
Does not use git mv (plot binaries are typically untracked).

Run from ANN-benchmark-HQ/:

    python scripts/migrate_qo_r095.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "analysis" / "plots"

QO_KEEP_FILES = {
    "all_query_results.csv",
    "all_query_results_gls_est.csv",
    "brute_force_recall_check.csv",
    "hyperparam_sweep.csv",
}
QO_KEEP_DIRS = {
    "gls_est",
    "runtime",
    "qps_recall_tertiles_per_algo",
    "r090",
    "r095",
    "r099",
}
QO_KEEP_PREFIXES = (
    "tertile_edges_",
)
# Threshold-dependent files that live at the workload root today (0.95 run).
QO_MOVE_EXACT = {
    "best_hyperparameters.csv",
    "hyperparam_recommendations.csv",
}
QO_MOVE_PREFIXES = (
    "best_plan_",
    "system_oracle_speedup.",
    "qps_recall_frontier_",
)
QO_MOVE_DIRS = ("optimal",)

PROTO_SKIP_DIRS = {"r090", "r095", "r099"}


def _is_qo_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("query_optimizer_")


def _is_proto_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("qo_prototype_")


def _should_keep_qo_file(name: str) -> bool:
    if name in QO_KEEP_FILES:
        return True
    return any(name.startswith(p) for p in QO_KEEP_PREFIXES)


def _should_move_qo_file(name: str) -> bool:
    if name in QO_MOVE_EXACT:
        return True
    return any(name.startswith(p) for p in QO_MOVE_PREFIXES)


def migrate_qo(workload: Path) -> str:
    dest = workload / "r095"
    sentinel = dest / "best_hyperparameters.csv"
    if sentinel.is_file():
        return f"skip {workload.name}: {sentinel.relative_to(PLOTS)} already exists"
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for child in list(workload.iterdir()):
        if child.name in QO_KEEP_DIRS:
            continue
        if child.is_dir() and child.name in QO_MOVE_DIRS:
            shutil.move(str(child), str(dest / child.name))
            moved += 1
            continue
        if child.is_file() and _should_move_qo_file(child.name):
            shutil.move(str(child), str(dest / child.name))
            moved += 1
            continue
        if child.is_file() and _should_keep_qo_file(child.name):
            continue
        # Leave anything else (stale mashup PNGs, logs) at the root.
    if moved == 0:
        try:
            dest.rmdir()
        except OSError:
            pass
        return f"skip {workload.name}: nothing threshold-dependent to move"
    return f"moved {moved} entries {workload.name}/ → {workload.name}/r095/"


def migrate_prototype(workload: Path) -> str:
    dest = workload / "r095"
    sentinel = dest / "gamma_pareto_exact.png"
    if sentinel.is_file():
        return f"skip {workload.name}: {sentinel.relative_to(PLOTS)} already exists"
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for child in list(workload.iterdir()):
        if child.name in PROTO_SKIP_DIRS:
            continue
        if child.is_file() or child.is_dir():
            shutil.move(str(child), str(dest / child.name))
            moved += 1
    sibling_log = PLOTS / f"{workload.name}.log"
    if sibling_log.is_file():
        shutil.move(str(sibling_log), str(dest / "run.log"))
        moved += 1
    return f"moved {moved} entries {workload.name}/ → {workload.name}/r095/"


def main() -> None:
    if not PLOTS.is_dir():
        raise SystemExit(f"missing {PLOTS}")
    reports = []
    for child in sorted(PLOTS.iterdir()):
        if _is_qo_dir(child):
            reports.append(migrate_qo(child))
        elif _is_proto_dir(child):
            reports.append(migrate_prototype(child))
    if not reports:
        print("No query_optimizer_* / qo_prototype_* directories found")
        return
    for line in reports:
        print(line)


if __name__ == "__main__":
    main()
