#!/usr/bin/env python3
"""Compute GLS-CorE ρ̂ for hard/superhard pack (q, filter) pairs.

The default estimator reports three nested neighborhoods — the nearest 1, 10,
and 100 nonempty IVF clusters — from one probe. A prefix larger than
``--sample-cap`` is a uniform sample, so local selectivity stays an estimate.
Pass ``--legacy-lsek`` to restore the single pooled neighborhood.

Writes::

    data/datasets/MoRe_large/stats/gls_est_{hard|superhard}_{movies|reviews}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_gls_core(glscore_root: Path):
    """Import the checkout next to this repo, not an older installed copy."""
    root = str(Path(glscore_root).resolve())
    sys.path.insert(0, root)
    for name in list(sys.modules):
        if name == "gls_core" or name.startswith("gls_core."):
            del sys.modules[name]
    import gls_core
    return gls_core


def pack_path(data_root: Path, scale: str, table: str, hardness: str) -> Path:
    if hardness == "hard":
        return data_root / f"MoRe_{scale}" / "hard_queries" / f"{table}_hcbgen_match_pdf.hdf5"
    return data_root / f"MoRe_{scale}" / "superhard_queries" / f"{table}_hcbgen_superhard.hdf5"


def estimate_pack(gc, est, ds, path: Path, table: str):
    """Tiered (or legacy) estimates for one hardness pack, reusing ``est``."""
    from gls_core.util import l2_normalize

    with h5py.File(path, "r") as f:
        Q = l2_normalize(f["test"].astype(np.float32)[...], copy=False)
        filters = [
            x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in f["filter"][:]
        ]
        sels = f["selectivity"][:].astype(np.float64)
    attr_map = {}
    for attr in (*ds.numeric_attrs, *ds.categorical_attrs):
        attr_map[attr] = attr
        attr_map[attr.replace("_", "").lower()] = attr
    specs = [
        gc.BenchmarkFilter(
            filter_id=0, raw=raw, selectivity=float(sel),
            filter=gc.parse_benchmark_filter(raw, attr_map),
        )
        for raw, sel in zip(filters, sels)
    ]
    df = gc.estimate_pairs(
        est, table, np.arange(len(Q)), Q, specs, verbose=True, aligned=True,
    )
    if len(df) != len(Q):
        raise RuntimeError(
            f"{path.name}: expected {len(Q)} aligned estimates, got {len(df)}"
        )
    return df


def _decode(values) -> np.ndarray:
    return np.array(
        [
            v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in np.asarray(values)
        ]
    )


def _zero_estimate_row(gc, query_index: int, raw: str, selectivity: float) -> dict:
    """sigma_g == 0 is outside the estimator. Record a neutral rho, matching the exact-GLS guard."""
    tiers = (1, 10, 100)
    row = {
        "q_id": int(query_index),
        "query_type": "cases",
        "filter_id": int(query_index),
        "filter": raw,
        "selectivity": float(selectivity),
        "estimator": "caselaw-zero-selectivity",
        "gls_correlation": 0.0,
        "sigma_l_hat": 0.0,
        "sigma_g": float(selectivity),
        "n_b": 0,
        "c": 0,
        "conf": 1.0,
        "capped": False,
        "latency_us": 0.0,
    }
    for n in tiers:
        row[f"gls_correlation_{n}"] = 0.0
        row[f"sigma_l_hat_{n}"] = float(selectivity)
        row[f"n_b_{n}"] = 0
        row[f"n_eval_{n}"] = 0
        row[f"c_{n}"] = 0
        row[f"sampled_{n}"] = False
    return row


def _read_h5_str_col(ds) -> np.ndarray:
    if hasattr(ds, "asstr"):
        return np.asarray(ds.asstr()[:], dtype=object)
    return _decode(ds[:])


def estimate_caselaw(gc, args) -> None:
    """Tiered rho-hat on the CaseLaw manifest population. Vectors stay unnormalized."""
    import json

    import pandas as pd

    try:
        import faiss

        faiss.omp_set_num_threads(48)
    except Exception:
        pass

    sys.path.insert(0, str(REPO_ROOT))
    from ann_benchmarks.corpora import cardinality, pack_root

    root = Path(pack_root("caselaw"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    manifest = json.loads((root / "sample_manifest.json").read_text(encoding="utf-8"))
    records = manifest["queries"]
    indexes = np.asarray([int(row["query_index"]) for row in records], dtype=np.int64)
    n_base = int(cardinality("caselaw", "cases"))

    base_path = root / "datasets" / "cases_dataset_0.hdf5"
    with h5py.File(base_path, "r") as f:
        # Euclidean pack: do not L2-normalize. MoRe hard estimation keeps its own call.
        X = np.asarray(f["train_vector"], dtype=np.float32)
        meta = {
            "date_filed": np.asarray(f["train_date_filed"], dtype=np.float64),
            "court_jurisdiction": _read_h5_str_col(f["train_court_jurisdiction"]),
            "court_type": _read_h5_str_col(f["train_court_type"]),
            "court_full_name": _read_h5_str_col(f["train_court_full_name"]),
        }
        ids = np.empty(0, dtype=object)
    if X.shape[0] != n_base:
        raise RuntimeError(f"{base_path} has {X.shape[0]} rows, registry N is {n_base}")

    ds = gc.Dataset(
        name="cases",
        X=X,
        ids=ids,
        meta=meta,
        numeric_attrs=("date_filed",),
        categorical_attrs=("court_jurisdiction", "court_type", "court_full_name"),
    )
    nlist = int(round(np.sqrt(n_base)))
    print(f"[cases] partitioner={args.partitioner} (nlist={nlist}) on N={n_base}, sample={len(indexes)}", flush=True)
    tiers = None if args.legacy_lsek else tuple(args.tiers)
    if args.partitioner == "pq_synopsis":
        caselaw_pparams = {"n_subspaces": 2, "tier_subcentroids": (64, 16, 4), "metric": "l2"}
    else:
        caselaw_pparams = {"nlist": nlist}
    est = gc.build_estimator(
        ds.X,
        ds.meta,
        ds.numeric_attrs,
        ds.categorical_attrs,
        partitioner=args.partitioner,
        partitioner_params=caselaw_pparams,
        lsek_min=args.lsek_min,
        shrink_a=args.shrink_a,
        tiers=tiers,
        sample_cap=args.sample_cap,
    )

    native = root / "native" / "cases_native.hdf5"
    with h5py.File(native, "r") as f:
        Q = np.asarray(f["test"][indexes], dtype=np.float32)
        filters = [
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in f["filter"][indexes]
        ]
        sels = np.asarray(f["selectivity"][indexes], dtype=np.float64)
    attr_map = {name: name for name in (*ds.numeric_attrs, *ds.categorical_attrs)}

    positive = [i for i, sel in enumerate(sels) if float(sel) > 0.0]
    frames = []
    if positive:
        pos = np.asarray(positive, dtype=np.int64)
        specs = [
            gc.BenchmarkFilter(
                filter_id=int(indexes[i]),
                raw=filters[i],
                selectivity=float(sels[i]),
                filter=gc.parse_benchmark_filter(filters[i], attr_map),
            )
            for i in positive
        ]
        estimated = gc.estimate_pairs(
            est, "cases", indexes[pos], Q[pos], specs, verbose=True, aligned=True,
        )
        frames.append(estimated)
    pos_set = set(positive)
    zero_rows = [
        _zero_estimate_row(gc, int(indexes[i]), filters[i], float(sels[i]))
        for i in range(len(indexes))
        if i not in pos_set
    ]
    if zero_rows:
        frames.append(pd.DataFrame(zero_rows))
    if not frames:
        raise RuntimeError("CaseLaw sample produced no GLS rows")
    df = pd.concat(frames, ignore_index=True)
    df["query_index"] = df["q_id"].astype(int)
    order = {int(q): pos for pos, q in enumerate(indexes.tolist())}
    df["_ord"] = df["query_index"].map(order)
    if df["_ord"].isna().any():
        raise RuntimeError("tiered estimates are missing a manifest query")
    df = df.sort_values("_ord").drop(columns="_ord")
    df["gls_correlation_est"] = df["gls_correlation"]
    df["q_id"] = df["query_index"].map(lambda q: f"qc{int(q):05d}")
    if len(df) != len(indexes):
        raise RuntimeError(f"expected {len(indexes)} estimates, got {len(df)}")

    out = root / "stats" / "gls_est_native_cases.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} estimates -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", default="large")
    ap.add_argument("--data-root", default="data/datasets")
    ap.add_argument("--corpus", choices=["morevec", "caselaw"], default="morevec")
    ap.add_argument("--hardness", choices=["hard", "superhard"], default=None)
    ap.add_argument("--tables", nargs="+", default=["movies", "reviews"])
    ap.add_argument("--partitioner", default="ivf")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--lsek-min", type=int, default=2048)
    ap.add_argument("--shrink-a", type=float, default=1.0)
    ap.add_argument("--sample-cap", type=int, default=200,
                    help="Filter checks per tier, split evenly across that tier's centroids")
    ap.add_argument("--tiers", type=int, nargs="*", default=[1, 10, 100],
                    help="Nonempty regions per nested GLS feature")
    ap.add_argument("--legacy-lsek", action="store_true",
                    help="Single lsek_min neighborhood instead of tiers")
    ap.add_argument("--glscore-root", default=str(REPO_ROOT.parent / "glscore"))
    args = ap.parse_args()

    gc = _import_gls_core(Path(args.glscore_root))
    if args.corpus == "caselaw":
        estimate_caselaw(gc, args)
        return
    if args.hardness is None:
        ap.error("--hardness is required for the MoReVec packs")

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    if args.partitioner == "pq_synopsis":
        pparams = {"n_subspaces": 2, "tier_subcentroids": (64, 16, 4), "metric": "cosine"}
    else:
        pparams = {"nlist": args.nlist}
    tiers = None if args.legacy_lsek else tuple(args.tiers)
    stats_dir = data_root / f"MoRe_{args.scale}" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    for table in args.tables:
        path = pack_path(data_root, args.scale, table, args.hardness)
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"[{table}/{args.hardness}] loading pack {path}", flush=True)
        print(f"[{table}] loading dataset + estimator...", flush=True)
        ds = gc.Dataset.load(data_root, table=table, scale=args.scale)
        est = gc.build_estimator(
            ds.X,
            ds.meta,
            ds.numeric_attrs,
            ds.categorical_attrs,
            partitioner=args.partitioner,
            partitioner_params=pparams,
            lsek_min=args.lsek_min,
            shrink_a=args.shrink_a,
            tiers=tiers,
            sample_cap=args.sample_cap,
        )
        df = estimate_pack(gc, est, ds, path, table)
        out = stats_dir / f"gls_est_{args.hardness}_{table}.csv"
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} estimates -> {out}", flush=True)


if __name__ == "__main__":
    main()
