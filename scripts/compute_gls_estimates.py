#!/usr/bin/env python3
"""Compute GLS correlation estimates for the benchmark workload via GLS-CorE.

This is the ETL driver that connects the two repos: it asks the ``gls_core``
package (the GLS-CorE repo) to estimate the GLS correlation rho_hat for every
(query vector, filter) pair of the MoReVec workload, then stores the result as
a single CSV in the dataset's ``stats/`` folder, ready for
``analysis/query_optimizer_analysis.py``.

It also (optionally) measures a per-table ANN-query latency baseline so the
estimator's runtime can be reported as a speedup ratio (the cost the estimator
*avoids* at planning time).

Run it from the ANN-Benchmarks repo root with the ``glscore`` conda env::

    conda activate glscore
    python scripts/compute_gls_estimates.py --scale large

Changing the estimator only changes flags here; the output schema (and the
downstream plots) stay fixed::

    python scripts/compute_gls_estimates.py --scale large \
        --partitioner ivf --nlist 4096 --lsek-min 2048 --shrink-a 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_gls_core(glscore_root: Path):
    """Import gls_core, falling back to a sibling checkout if not installed."""
    try:
        import gls_core  # noqa: F401
        return gls_core
    except ModuleNotFoundError:
        if glscore_root.is_dir():
            sys.path.insert(0, str(glscore_root))
        import gls_core  # noqa: F401
        return gls_core


def measure_runtime_baseline(gc, data_root: Path, scale: str,
                             tables: list[str], df, est_name: str,
                             n_baseline_queries: int) -> dict:
    """ANN-query latency baseline per table + estimator latency summary.

    The estimator predicts rho_hat *without* a vector search; the natural cost
    yardstick is therefore a single exact-ish ANN query on the same data
    (plan section 4.2). We report estimator latency as a ratio against it.
    """
    from gls_core import eval as gls_eval

    meta = {"estimator": est_name, "tables": {}}
    for table in tables:
        qt = gc.TABLE_QUERY_TYPE[table]
        est_lat = df.loc[df["query_type"] == qt, "latency_us"]
        est_lat = est_lat[est_lat > 0]  # drop the No_filter (0) rows
        ds = gc.Dataset.load(data_root, table=table, scale=scale)
        _, Q = gc.load_benchmark_queries(data_root, table, scale)
        Q = Q[:n_baseline_queries]
        print(f"[baseline] {table}: ANN query latency over {len(Q)} queries "
              f"(N={ds.N})...", flush=True)
        ann = gls_eval.ann_query_latency(ds.X, Q)
        ann_us = float(ann["mean_latency_us"])
        est_mean = float(est_lat.mean())
        meta["tables"][table] = {
            "query_type": qt,
            "ann_baseline_latency_us": ann_us,
            "ann_baseline_cfg": {k: ann[k] for k in
                                 ("k", "nlist", "nprobe", "n_queries")},
            "estimator_mean_latency_us": est_mean,
            "estimator_p50_latency_us": float(est_lat.quantile(0.50)),
            "estimator_p99_latency_us": float(est_lat.quantile(0.99)),
            "speedup_vs_ann": ann_us / est_mean if est_mean > 0 else None,
        }
        print(f"[baseline] {table}: ANN {ann_us:.1f} us vs estimator "
              f"{est_mean:.1f} us  ->  {ann_us / est_mean:.1f}x cheaper",
              flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="large",
                    choices=("small", "medium", "large"))
    ap.add_argument("--data-root", default="data/datasets",
                    help="dataset root (relative to repo root or absolute)")
    ap.add_argument("--tables", nargs="+", default=["movies", "reviews"],
                    choices=("movies", "reviews"))
    ap.add_argument("--partitioner", default="ivf",
                    choices=("ivf", "simhash"))
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--n-bits", type=int, default=12)
    ap.add_argument("--n-tables", type=int, default=4)
    ap.add_argument("--lsek-min", type=int, default=2048)
    ap.add_argument("--shrink-a", type=float, default=1.0)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="output CSV path (default: stats/gls_correlation_"
                         "estimates_<kind>_lsek<L>_a<a>_<mode>.csv)")
    ap.add_argument("--ann-baseline", action="store_true", default=True,
                    help="measure ANN-query latency baseline (default on)")
    ap.add_argument("--no-ann-baseline", dest="ann_baseline",
                    action="store_false")
    ap.add_argument("--ann-baseline-queries", type=int, default=200)
    ap.add_argument("--glscore-root", default=str(REPO_ROOT.parent / "glscore"),
                    help="path to the glscore checkout (import fallback)")
    args = ap.parse_args()

    gc = _import_gls_core(Path(args.glscore_root))

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    if args.partitioner == "ivf":
        pparams = {"nlist": args.nlist}
    else:
        pparams = {"n_bits": args.n_bits, "n_tables": args.n_tables}

    print("=" * 70)
    print("Computing GLS correlation estimates")
    print("=" * 70)
    print(f"  data root : {data_root}")
    print(f"  scale     : {args.scale}")
    print(f"  tables    : {args.tables}")
    print(f"  estimator : {args.partitioner} {pparams} "
          f"lsek_min={args.lsek_min} shrink_a={args.shrink_a}")
    print()

    df = gc.compute_all_tables(
        data_root, args.scale, tables=tuple(args.tables),
        partitioner=args.partitioner, partitioner_params=pparams,
        lsek_min=args.lsek_min, shrink_a=args.shrink_a,
        mode=args.mode, variant=args.variant,
    )

    if args.output is None:
        out_path = gc.default_estimates_path(
            data_root, args.scale, partitioner=args.partitioner,
            lsek_min=args.lsek_min, shrink_a=args.shrink_a, mode=args.mode)
    else:
        out_path = Path(args.output)
    gc.write_gls_estimates(df, out_path)
    print(f"\nSaved {len(df)} estimates -> {out_path}")
    est_name = df["estimator"].iloc[0]

    if args.ann_baseline:
        print("\n" + "=" * 70)
        print("Runtime baseline (ANN query latency)")
        print("=" * 70)
        try:
            meta = measure_runtime_baseline(
                gc, data_root, args.scale, args.tables, df, est_name,
                args.ann_baseline_queries)
            baseline_path = out_path.with_name(
                out_path.stem + "_runtime_baseline.json")
            baseline_path.write_text(json.dumps(meta, indent=2))
            print(f"\nSaved runtime baseline -> {baseline_path}")
        except Exception as exc:  # ANN baseline is best-effort
            print(f"[baseline] skipped ({type(exc).__name__}: {exc})")

    print("\nDone.")


if __name__ == "__main__":
    main()
