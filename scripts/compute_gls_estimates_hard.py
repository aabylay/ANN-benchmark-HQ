#!/usr/bin/env python3
"""Compute GLS-CorE ρ̂ for hard/superhard pack (q, filter) pairs.

Uses the same estimator settings as the flex large MoRe run
(``ivf``, ``nlist=1024``, ``lsek_min=2048``, ``shrink_a=1.0``).

Writes::

    data/datasets/MoRe_large/stats/gls_est_{hard|superhard}_{movies|reviews}.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_gls_core(glscore_root: Path):
    try:
        import gls_core  # noqa: F401
        return gls_core
    except ModuleNotFoundError:
        sys.path.insert(0, str(glscore_root))
        import gls_core  # noqa: F401
        return gls_core


def pack_path(data_root: Path, scale: str, table: str, hardness: str) -> Path:
    if hardness == "hard":
        return data_root / f"MoRe_{scale}" / "hard_queries" / f"{table}_hcbgen_match_pdf.hdf5"
    return data_root / f"MoRe_{scale}" / "superhard_queries" / f"{table}_hcbgen_superhard.hdf5"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", default="large")
    ap.add_argument("--data-root", default="data/datasets")
    ap.add_argument("--hardness", required=True, choices=["hard", "superhard"])
    ap.add_argument("--tables", nargs="+", default=["movies", "reviews"])
    ap.add_argument("--partitioner", default="ivf")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--lsek-min", type=int, default=2048)
    ap.add_argument("--shrink-a", type=float, default=1.0)
    ap.add_argument("--glscore-root", default=str(REPO_ROOT.parent / "glscore"))
    args = ap.parse_args()

    gc = _import_gls_core(Path(args.glscore_root))
    from gls_core.util import l2_normalize

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    pparams = {"nlist": args.nlist}
    stats_dir = data_root / f"MoRe_{args.scale}" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    for table in args.tables:
        path = pack_path(data_root, args.scale, table, args.hardness)
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"[{table}/{args.hardness}] loading pack {path}", flush=True)
        with h5py.File(path, "r") as f:
            Q = l2_normalize(f["test"].astype(np.float32)[...], copy=False)
            filters = [
                x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in f["filter"][:]
            ]
            sels = f["selectivity"][:].astype(np.float64)

        print(f"[{table}] loading dataset + estimator...", flush=True)
        ds = gc.Dataset.load(data_root, table=table, scale=args.scale)
        attr_map = {}
        for attr in (*ds.numeric_attrs, *ds.categorical_attrs):
            attr_map[attr] = attr
            attr_map[attr.replace("_", "").lower()] = attr

        est = gc.build_estimator(
            ds.X,
            ds.meta,
            ds.numeric_attrs,
            ds.categorical_attrs,
            partitioner=args.partitioner,
            partitioner_params=pparams,
            lsek_min=args.lsek_min,
            shrink_a=args.shrink_a,
        )

        rows = []
        for qi, (q, raw, sel) in enumerate(zip(Q, filters, sels)):
            flt = gc.parse_benchmark_filter(raw, attr_map)
            base = {
                "q_id": qi,
                "query_type": table,
                "filter_id": 0,
                "filter": raw,
                "selectivity": float(sel),
                "estimator": est.name,
            }
            if flt is None:
                rows.append(
                    {
                        **base,
                        "gls_correlation": 0.0,
                        "sigma_l_hat": float(sel),
                        "sigma_g": float(sel),
                        "n_b": 0,
                        "c": 0,
                        "conf": 1.0,
                        "capped": False,
                        "latency_us": 0.0,
                    }
                )
                continue
            sel_b = est.select_buckets(q)
            t0 = time.perf_counter()
            r = est.estimate(q, flt, float(sel), selection=sel_b)
            us = (time.perf_counter() - t0) * 1e6 + sel_b.probe_seconds * 1e6
            rows.append(
                {
                    **base,
                    "gls_correlation": float(r.rho_hat),
                    "sigma_l_hat": float(r.sigma_l_hat),
                    "sigma_g": float(r.sigma_g),
                    "n_b": r.n_b,
                    "c": r.c,
                    "conf": r.conf,
                    "capped": r.capped,
                    "latency_us": us,
                }
            )
            if (qi + 1) % 100 == 0:
                print(f"  {qi+1}/{len(Q)}", flush=True)

        df = pd.DataFrame(rows)
        out = stats_dir / f"gls_est_{args.hardness}_{table}.csv"
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} estimates -> {out}", flush=True)


if __name__ == "__main__":
    main()
