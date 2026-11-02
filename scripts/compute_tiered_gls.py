#!/usr/bin/env python3
"""Fit one tiered GLS estimator per table and write every workload that uses it.

Neighborhood radii are the nearest 1, 10, and 100 nonempty partitions. The
probe is done once per query; each radius is a prefix, sampled when it holds
more than --sample-cap distinct vectors. Outputs:

  stats/gls_correlation_estimates_{kind}_tiers1-10-100_cap{cap}_a1_0.csv
  stats/gls_est_{hard,superhard}_{movies,reviews}.csv

The previous single-neighborhood hard/superhard CSVs are copied to
``gls_est_*_lsek2048.csv`` before being replaced.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_gls_estimates_hard import (  # noqa: E402
    estimate_pack,
    pack_path,
    _import_gls_core,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", default="large")
    ap.add_argument("--data-root", default="data/datasets")
    ap.add_argument("--tables", nargs="+", default=["movies", "reviews"])
    ap.add_argument("--partitioner", default="pq_synopsis")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--n-bits", type=int, default=12)
    ap.add_argument("--n-tables", type=int, default=4)
    ap.add_argument("--shrink-a", type=float, default=1.0)
    ap.add_argument("--sample-cap", type=int, default=200,
                    help="Filter checks per tier, split evenly across that tier's centroids")
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--workloads", nargs="+",
                    default=["flex", "hard", "superhard"])
    ap.add_argument("--glscore-root", default=str(REPO_ROOT.parent / "glscore"))
    args = ap.parse_args()

    gc = _import_gls_core(Path(args.glscore_root))
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    tiers = tuple(args.tiers)
    if args.partitioner == "pq_synopsis":
        pparams = {"n_subspaces": 2, "tier_subcentroids": (64, 16, 4)}
    elif args.partitioner == "ivf":
        pparams = {"nlist": args.nlist}
    else:
        pparams = {"n_bits": args.n_bits, "n_tables": args.n_tables}

    stats_dir = data_root / f"MoRe_{args.scale}" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for hardness in ("hard", "superhard"):
        for table in args.tables:
            src = stats_dir / f"gls_est_{hardness}_{table}.csv"
            backup = stats_dir / f"gls_est_{hardness}_{table}_lsek2048.csv"
            if src.is_file() and not backup.is_file():
                shutil.copy2(src, backup)
                print(f"backed up {src.name} -> {backup.name}", flush=True)

    for table in args.tables:
        print(f"\n=== fit {args.partitioner} {pparams} on {table} ===", flush=True)
        ds = gc.Dataset.load(data_root, table=table, scale=args.scale)
        est = gc.build_estimator(
            ds.X, ds.meta, ds.numeric_attrs, ds.categorical_attrs,
            partitioner=args.partitioner, partitioner_params=pparams,
            shrink_a=args.shrink_a, tiers=tiers, sample_cap=args.sample_cap,
        )
        print(f"estimator {est.name}", flush=True)

        if "flex" in args.workloads:
            df = gc.compute_gls_estimates(
                data_root, table, args.scale,
                partitioner=args.partitioner, partitioner_params=pparams,
                shrink_a=args.shrink_a, tiers=tiers, sample_cap=args.sample_cap,
                estimator=est, dataset=ds,
            )
            out = gc.default_estimates_path(
                data_root, args.scale, partitioner=args.partitioner,
                shrink_a=args.shrink_a, tiers=tiers, sample_cap=args.sample_cap,
            )
            # compute_gls_estimates is per table; concatenate on disk.
            part = stats_dir / f".tiered_{table}.csv"
            df.to_csv(part, index=False)
            print(f"flex {table}: {len(df)} rows -> {part}", flush=True)

        for hardness in ("hard", "superhard"):
            if hardness not in args.workloads:
                continue
            path = pack_path(data_root, args.scale, table, hardness)
            print(f"[{table}/{hardness}] {path}", flush=True)
            df = estimate_pack(gc, est, ds, path, table)
            out = stats_dir / f"gls_est_{hardness}_{table}.csv"
            df.to_csv(out, index=False)
            print(f"saved {len(df)} -> {out}", flush=True)

        del est, ds

    if "flex" in args.workloads:
        import pandas as pd
        parts = [stats_dir / f".tiered_{table}.csv" for table in args.tables]
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        out = gc.default_estimates_path(
            data_root, args.scale, partitioner=args.partitioner,
            shrink_a=args.shrink_a, tiers=tiers, sample_cap=args.sample_cap,
        )
        gc.write_gls_estimates(df, out)
        print(f"\nflex combined {len(df)} -> {out}", flush=True)
        if args.partitioner == "pq_synopsis":
            compat = gc.default_estimates_path(
                data_root, args.scale, partitioner="ivf",
                shrink_a=args.shrink_a, tiers=tiers, sample_cap=args.sample_cap,
            )
            if compat != out:
                gc.write_gls_estimates(df, compat)
                print(f"flex compatibility copy -> {compat}", flush=True)


if __name__ == "__main__":
    main()
