#!/usr/bin/env python3
"""Download and build the full single-vector CaseLaw_7M pack (N = 7,414,023).

Usage (from ANN-benchmark-HQ):
    python datasets_etl/caselaw_pack.py
    python datasets_etl/caselaw_pack.py --qo-sample-frac 0.10
    python datasets_etl/caselaw_pack.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ann_benchmarks.attrs import (
    _get_cat_date_index,
    _get_cat_index,
    _get_cat_pair_index,
    _get_date_index,
    canonicalize_attr_name,
    parse_predicate,
    predicate_mask,
)
from ann_benchmarks.corpora import (
    CASELAW_N_BASE,
    CASELAW_N_QUERIES_OFFICIAL,
    DEFAULT_QO_SAMPLE_FRAC,
    DEFAULT_SAMPLE_SEED,
    build_sample_manifest,
    canonical_filter_from_json,
    dump_manifest_bytes,
    pack_root,
    validate_qo_sample_frac,
)

UPSTREAM_COMMIT = "89a3abaafa63dda46b94b308bdf039e699841b3b"
BASE_URL = "https://comp21storage.z5.web.core.windows.net/caselaw/single_vector/"

# Recorded upstream byte sizes for caselaw-7M (full single-vector base + queries + GT + metadata).
EXPECTED_RAW_FILES: Dict[str, int] = {
    "caselaw_base_embeddings.bin": 45_551_757_320,
    "caselaw_query_embeddings.bin": 551_804_936,
    "caselaw_gt.bin": 71_849_608,
    "caselaw_gt_filtered.bin": 36_162_440,
    "caselaw_base_metadata.jsonl": 1_301_991_216,
    "caselaw_query_metadata.jsonl": 15_595_530,
    "caselaw_query_filters.jsonl": 11_849_200,
}

K_GT = 100


def source_urls() -> List[str]:
    return [BASE_URL + name for name in EXPECTED_RAW_FILES]


def _download_file_parallel(
    url: str,
    dest: Path,
    expected_size: int,
    workers: int = 16,
    chunk_size: int = 32 * 1024 * 1024,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expected_size:
        print(f"[download] skip {dest.name} (matches {expected_size:,} bytes)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    print(f"[download] fetching {url} -> {dest} ({expected_size:,} bytes)...", flush=True)
    fd = os.open(str(tmp), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    os.ftruncate(fd, expected_size)
    ranges: List[Tuple[int, int]] = []
    for start in range(0, expected_size, chunk_size):
        end = min(start + chunk_size - 1, expected_size - 1)
        ranges.append((start, end))

    done_chunks = [0]

    def fetch_range(rng: Tuple[int, int]) -> None:
        start, end = rng
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "Mozilla/5.0",
                    },
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    pos = start
                    while True:
                        buf = resp.read(1024 * 1024)
                        if not buf:
                            break
                        os.pwrite(fd, buf, pos)
                        pos += len(buf)
                    if pos != end + 1:
                        raise RuntimeError(f"Short read on {dest.name}: {pos} vs {end + 1}")
                done_chunks[0] += 1
                if done_chunks[0] % 100 == 0 or done_chunks[0] == len(ranges):
                    elapsed = time.time() - t0
                    print(
                        f"  [{dest.name}] {done_chunks[0]}/{len(ranges)} chunks ({elapsed:.1f}s)",
                        flush=True,
                    )
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(fetch_range, ranges))
    finally:
        os.close(fd)

    actual = tmp.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"{tmp} size {actual} != expected {expected_size}")
    tmp.replace(dest)
    dt = max(time.time() - t0, 1e-3)
    print(
        f"[download] finished {dest.name} in {dt:.1f}s ({expected_size / 1e6 / dt:.1f} MB/s)",
        flush=True,
    )


def download_raw_files(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_size in EXPECTED_RAW_FILES.items():
        _download_file_parallel(BASE_URL + filename, raw_dir / filename, expected_size)


def verify_raw_files_present(raw_dir: Path) -> None:
    for filename, expected_size in EXPECTED_RAW_FILES.items():
        p = raw_dir / filename
        if not p.is_file():
            raise FileNotFoundError(f"Missing raw file: {p}")
        actual = p.stat().st_size
        if actual != expected_size:
            raise RuntimeError(
                f"Raw file {p} has size {actual:,}, expected {expected_size:,}"
            )


def read_fbin_header(path: Path) -> Tuple[int, int]:
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise RuntimeError(f"{path}: truncated .fbin header")
    n, d = struct.unpack("<ii", header)
    expected_bytes = 8 + int(n) * int(d) * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path}: file size {actual_bytes:,} != 8 + {n}*{d}*4 ({expected_bytes:,})"
        )
    return int(n), int(d)


def read_fbin(path: Path) -> np.ndarray:
    n, d = read_fbin_header(path)
    arr = np.fromfile(path, dtype="<f4", count=n * d, offset=8)
    if arr.size != n * d:
        raise RuntimeError(f"{path}: read {arr.size} floats, expected {n * d}")
    return arr.reshape(n, d)


def read_knn_gt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read standard big-ann k-NN ground truth (uint32 n, k; int32 ids; float32 dists)."""
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"{path}: truncated knn GT header")
        n, k = struct.unpack("<II", header)
        n, k = int(n), int(k)
        expected = 8 + n * k * 8
        actual = path.stat().st_size
        if actual != expected:
            raise RuntimeError(f"{path}: size {actual} != expected {expected}")
        ids = np.fromfile(f, dtype="<i4", count=n * k).reshape(n, k)
        dists = np.fromfile(f, dtype="<f4", count=n * k).reshape(n, k)
    return ids, dists


def read_range_gt(path: Path, k_pad: int = K_GT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read big-ann range format filtered ground truth.

    Returns (nres[nq], gt_ids[nq, k_pad] padded with -1, gt_dists[nq, k_pad] padded with +inf).
    """
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"{path}: truncated range GT header")
        nq, total_res = struct.unpack("<ii", header)
        nq, total_res = int(nq), int(total_res)
        nres = np.fromfile(f, dtype="<i4", count=nq)
        if len(nres) != nq:
            raise RuntimeError(f"{path}: expected {nq} nres entries, got {len(nres)}")
        if int(nres.sum()) != total_res:
            raise RuntimeError(
                f"{path}: sum(nres)={int(nres.sum())} != total_res={total_res}"
            )
        flat_ids = np.fromfile(f, dtype="<i4", count=total_res)
        if len(flat_ids) != total_res:
            raise RuntimeError(f"{path}: expected {total_res} ids, got {len(flat_ids)}")
        flat_dists = np.fromfile(f, dtype="<f4", count=total_res)

    gt_ids = np.full((nq, k_pad), -1, dtype=np.int32)
    gt_dists = np.full((nq, k_pad), np.inf, dtype=np.float32)
    has_dists = len(flat_dists) == total_res
    offsets = np.zeros(nq + 1, dtype=np.int64)
    np.cumsum(nres, out=offsets[1:])
    for i in range(nq):
        cnt = int(nres[i])
        if cnt <= 0:
            continue
        take = min(cnt, k_pad)
        s = int(offsets[i])
        gt_ids[i, :take] = flat_ids[s : s + take]
        if has_dists:
            gt_dists[i, :take] = flat_dists[s : s + take]
    return nres, gt_ids, gt_dists


def load_query_filters_and_check_metadata(raw_dir: Path) -> List[str]:
    filters_path = raw_dir / "caselaw_query_filters.jsonl"
    qmeta_path = raw_dir / "caselaw_query_metadata.jsonl"

    filters: List[str] = []
    with open(filters_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if i == 0:
                print(f"[align] first filter JSON record: {obj}", flush=True)
            if int(obj["query_id"]) != i:
                raise RuntimeError(
                    f"{filters_path}: line {i} has query_id={obj.get('query_id')}"
                )
            filters.append(canonical_filter_from_json(obj["filter"]))
    if len(filters) != CASELAW_N_QUERIES_OFFICIAL:
        raise RuntimeError(
            f"{filters_path}: expected {CASELAW_N_QUERIES_OFFICIAL} lines, got {len(filters)}"
        )

    qmeta_count = 0
    with open(qmeta_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if int(obj["doc_id"]) != i:
                raise RuntimeError(
                    f"{qmeta_path}: line {i} has doc_id={obj.get('doc_id')}"
                )
            qmeta_count += 1
    if qmeta_count != CASELAW_N_QUERIES_OFFICIAL:
        raise RuntimeError(
            f"{qmeta_path}: expected {CASELAW_N_QUERIES_OFFICIAL} lines, got {qmeta_count}"
        )
    return filters


def load_base_metadata(raw_dir: Path) -> Dict[str, np.ndarray]:
    bmeta_path = raw_dir / "caselaw_base_metadata.jsonl"
    n = CASELAW_N_BASE
    print(f"[pack] reading {bmeta_path} ({n:,} lines)...", flush=True)
    t0 = time.time()

    train_id = np.empty(n, dtype=object)
    train_case_id = np.empty(n, dtype=object)
    train_date_filed = np.empty(n, dtype=np.int64)
    train_court_jurisdiction = np.empty(n, dtype=object)
    train_court_type = np.empty(n, dtype=object)
    train_court_full_name = np.empty(n, dtype=object)

    count = 0
    with open(bmeta_path, "r", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        for i, line in enumerate(f):
            if i >= n:
                count = i + 1
                break
            obj = json.loads(line)
            doc_id = int(obj["doc_id"])
            if doc_id != i:
                raise RuntimeError(f"{bmeta_path}: line {i} has doc_id={doc_id}")
            train_id[i] = str(doc_id)
            train_case_id[i] = str(obj["case_id"])
            train_date_filed[i] = int(obj["date_filed"])
            train_court_jurisdiction[i] = str(obj["court_jurisdiction"])
            train_court_type[i] = str(obj["court_type"])
            train_court_full_name[i] = str(obj["court_full_name"])
            count = i + 1
            if count % 1_000_000 == 0:
                print(f"  [base_metadata] {count:,}/{n:,} ({time.time() - t0:.1f}s)", flush=True)

    if count != n:
        raise RuntimeError(f"{bmeta_path}: expected {n} lines, got {count}")
    print(f"[pack] loaded {n:,} base metadata rows in {time.time() - t0:.1f}s", flush=True)
    return {
        "train_id": train_id,
        "train_case_id": train_case_id,
        "train_date_filed": train_date_filed,
        "train_court_jurisdiction": train_court_jurisdiction,
        "train_court_type": train_court_type,
        "train_court_full_name": train_court_full_name,
    }


def _base_hdf5_valid(base_hdf5: Path, expected_d: int) -> bool:
    if not base_hdf5.is_file():
        return False
    try:
        with h5py.File(base_hdf5, "r") as f:
            if int(f.attrs.get("dimension", -1)) != expected_d:
                return False
            if str(f.attrs.get("metric", "")) != "euclidean":
                return False
            req = (
                "train_vector",
                "train_id",
                "train_case_id",
                "train_date_filed",
                "train_court_jurisdiction",
                "train_court_type",
                "train_court_full_name",
            )
            for k in req:
                if k not in f or f[k].shape[0] != CASELAW_N_BASE:
                    return False
            if f["train_vector"].shape != (CASELAW_N_BASE, expected_d):
                return False
        return True
    except Exception:
        return False


def write_base_hdf5(
    base_hdf5: Path,
    base_fbin: Path,
    base_meta: Mapping[str, np.ndarray],
    n: int,
    d: int,
) -> None:
    base_hdf5.parent.mkdir(parents=True, exist_ok=True)
    tmp = base_hdf5.with_suffix(".hdf5.part")
    if tmp.exists():
        tmp.unlink()
    dt_str = h5py.string_dtype(encoding="utf-8")
    t0 = time.time()
    print(f"[pack] writing {base_hdf5} (N={n:,}, d={d})...", flush=True)
    with h5py.File(tmp, "w") as h5f:
        h5f.attrs["dimension"] = int(d)
        h5f.attrs["metric"] = "euclidean"
        h5f.attrs["point_type"] = "float"
        h5f.attrs["N"] = int(n)

        vec_ds = h5f.create_dataset("train_vector", shape=(n, d), dtype=np.float32)
        chunk_rows = 250_000
        with open(base_fbin, "rb") as bf:
            bf.seek(8)
            for start in range(0, n, chunk_rows):
                rows = min(chunk_rows, n - start)
                arr = np.fromfile(bf, dtype="<f4", count=rows * d)
                if arr.size != rows * d:
                    raise RuntimeError(
                        f"{base_fbin}: short read at row {start} ({arr.size} vs {rows * d})"
                    )
                vec_ds[start : start + rows] = arr.reshape(rows, d)
                print(
                    f"  [train_vector] {start + rows:,}/{n:,} ({time.time() - t0:.1f}s)",
                    flush=True,
                )

        if vec_ds.shape[1] != d:
            raise RuntimeError(f"Header dimension {d} != train_vector.shape[1] {vec_ds.shape[1]}")

        h5f.create_dataset("train_date_filed", data=base_meta["train_date_filed"], dtype=np.int64)
        for key in (
            "train_id",
            "train_case_id",
            "train_court_jurisdiction",
            "train_court_type",
            "train_court_full_name",
        ):
            t_col = time.time()
            h5f.create_dataset(key, data=base_meta[key], dtype=dt_str)
            print(f"  [{key}] written in {time.time() - t_col:.1f}s", flush=True)

    tmp.replace(base_hdf5)
    print(f"[pack] wrote {base_hdf5} in {time.time() - t0:.1f}s", flush=True)


def load_attrs_from_base_hdf5(base_hdf5: Path) -> Dict[str, np.ndarray]:
    print(f"[pack] loading metadata columns from {base_hdf5}...", flush=True)
    t0 = time.time()
    with h5py.File(base_hdf5, "r") as f:
        date_filed = np.asarray(f["train_date_filed"][:], dtype=np.int64)
        court_jurisdiction = np.asarray(f["train_court_jurisdiction"].asstr()[:], dtype=object)
        court_type = np.asarray(f["train_court_type"].asstr()[:], dtype=object)
        court_full_name = np.asarray(f["train_court_full_name"].asstr()[:], dtype=object)
    print(f"[pack] loaded metadata columns in {time.time() - t0:.1f}s", flush=True)
    return {
        "date_filed": date_filed,
        "court_jurisdiction": court_jurisdiction,
        "court_type": court_type,
        "court_full_name": court_full_name,
    }


def _count_passing_fast(attrs: Mapping[str, np.ndarray], clauses: Sequence[Tuple[str, str, str]], n: int) -> int:
    cat_eq: List[Tuple[str, str]] = []
    date_ge: List[int] = []
    date_le: List[int] = []
    for raw_attr, op, val in clauses:
        canon = canonicalize_attr_name(raw_attr)
        if canon == "date_filed":
            ival = int(val)
            if op == ">=":
                date_ge.append(ival)
            elif op == "<=":
                date_le.append(ival)
            else:
                return int(predicate_mask(attrs, clauses).sum())
        elif op == "=":
            cat_eq.append((canon, str(val)))
        else:
            return int(predicate_mask(attrs, clauses).sum())

    lo_val = max(date_ge) if date_ge else None
    hi_val = min(date_le) if date_le else None
    if lo_val is not None and hi_val is not None and lo_val > hi_val:
        return 0

    date_arr = attrs["date_filed"]
    if len(cat_eq) == 1 and (lo_val is not None or hi_val is not None):
        cat_name, cat_val = cat_eq[0]
        bounds, s_dates, _ = _get_cat_date_index(attrs[cat_name], date_arr)
        span = bounds.get(cat_val)
        if span is None:
            return 0
        d_slice = s_dates[span[0] : span[1]]
        i_lo = int(np.searchsorted(d_slice, lo_val, side="left")) if lo_val is not None else 0
        i_hi = int(np.searchsorted(d_slice, hi_val, side="right")) if hi_val is not None else len(d_slice)
        return max(0, i_hi - i_lo)

    if len(cat_eq) == 1 and lo_val is None and hi_val is None:
        cat_name, cat_val = cat_eq[0]
        bounds, _ = _get_cat_index(attrs[cat_name])
        span = bounds.get(cat_val)
        return 0 if span is None else (span[1] - span[0])

    if len(cat_eq) == 2 and lo_val is None and hi_val is None:
        (c1, v1), (c2, v2) = cat_eq
        bounds, _ = _get_cat_pair_index(attrs[c1], attrs[c2])
        span = bounds.get((v1, v2))
        return 0 if span is None else (span[1] - span[0])

    if len(cat_eq) == 0 and (lo_val is not None or hi_val is not None):
        s_dates, _ = _get_date_index(date_arr)
        i_lo = int(np.searchsorted(s_dates, lo_val, side="left")) if lo_val is not None else 0
        i_hi = int(np.searchsorted(s_dates, hi_val, side="right")) if hi_val is not None else len(s_dates)
        return max(0, i_hi - i_lo)

    return int(predicate_mask(attrs, clauses).sum())


def compute_all_selectivities(
    attrs: Mapping[str, np.ndarray],
    filters: Sequence[str],
    n_base: int = CASELAW_N_BASE,
) -> np.ndarray:
    t0 = time.time()
    print(f"[pack] computing selectivities for {len(filters):,} queries on N={n_base:,}...", flush=True)
    sels = np.empty(len(filters), dtype=np.float64)
    cache: Dict[str, float] = {}
    for i, fstr in enumerate(filters):
        val = cache.get(fstr)
        if val is None:
            clauses = parse_predicate(fstr)
            cnt = _count_passing_fast(attrs, clauses, n_base)
            val = float(cnt) / float(n_base)
            cache[fstr] = val
        sels[i] = val

    # Sanity-check 10 queries against full boolean predicate_mask(attrs, ...).mean()
    for check_i in range(min(10, len(filters))):
        expected = float(predicate_mask(attrs, parse_predicate(filters[check_i])).mean())
        if not np.isclose(sels[check_i], expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"Selectivity mismatch at query {check_i}: fast={sels[check_i]} vs mask={expected}"
            )
    print(f"[pack] computed {len(filters):,} selectivities in {time.time() - t0:.2f}s", flush=True)
    return sels


def run_metric_check(
    base_hdf5: Path,
    queries: np.ndarray,
    nres: np.ndarray,
    nn_ids: np.ndarray,
    nn_dists: np.ndarray,
    out_json: Path,
) -> None:
    """Section 8: compare first published unfiltered neighbor distance on 20 queries (seed 42)."""
    eligible = np.flatnonzero(nres >= 1)
    chosen = np.sort(
        np.random.RandomState(42).choice(eligible, size=20, replace=False)
    )
    pairs = []
    rel_l2 = []
    rel_l2_sq = []
    with h5py.File(base_hdf5, "r") as f:
        vec_ds = f["train_vector"]
        for qi in chosen:
            qi_int = int(qi)
            nid = int(nn_ids[qi_int, 0])
            pub = float(nn_dists[qi_int, 0])
            q = queries[qi_int].astype(np.float64)
            x = np.asarray(vec_ds[nid], dtype=np.float64)
            diff = q - x
            l2_sq = float(np.dot(diff, diff))
            l2 = float(np.sqrt(max(l2_sq, 0.0)))
            denom = max(abs(pub), 1e-12)
            e_l2 = abs(l2 - pub) / denom
            e_sq = abs(l2_sq - pub) / denom
            rel_l2.append(e_l2)
            rel_l2_sq.append(e_sq)
            pairs.append(
                {
                    "query_index": qi_int,
                    "neighbor_id": nid,
                    "published_distance": pub,
                    "l2": l2,
                    "l2_squared": l2_sq,
                    "rel_error_l2": e_l2,
                    "rel_error_l2_squared": e_sq,
                }
            )

    med_l2 = float(np.median(rel_l2))
    med_sq = float(np.median(rel_l2_sq))
    print(
        f"[metric_check] median rel error: L2={med_l2:.3e}, L2^2={med_sq:.3e}",
        flush=True,
    )
    if min(med_l2, med_sq) >= 1e-3:
        raise RuntimeError(
            f"Metric check failed: median rel error L2={med_l2}, L2^2={med_sq} (both >= 1e-3)"
        )
    payload = {
        "metric": "euclidean",
        "seed": 42,
        "n_checked": 20,
        "median_rel_error_l2": med_l2,
        "median_rel_error_l2_squared": med_sq,
        "pairs": pairs,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_membership_check(
    attrs: Mapping[str, np.ndarray],
    filters: Sequence[str],
    nres: np.ndarray,
    gt_ids: np.ndarray,
    n_base: int = CASELAW_N_BASE,
) -> None:
    """Section 9: 50 queries with nres >= 1 (seed 42); every published id in [0, N) and satisfies predicate."""
    eligible = np.flatnonzero(nres >= 1)
    chosen = np.sort(
        np.random.RandomState(42).choice(eligible, size=50, replace=False)
    )
    for qi in chosen:
        qi_int = int(qi)
        cnt = min(int(nres[qi_int]), gt_ids.shape[1])
        ids = gt_ids[qi_int, :cnt]
        if np.any(ids < 0) or np.any(ids >= n_base):
            raise RuntimeError(
                f"Membership check failed on query {qi_int}: id out of [0, {n_base})"
            )
        sub_attrs = {k: np.asarray(v)[ids] for k, v in attrs.items()}
        mask = predicate_mask(sub_attrs, parse_predicate(filters[qi_int]))
        if not bool(np.all(mask)):
            bad = ids[~mask].tolist()
            raise RuntimeError(
                f"Membership check failed on query {qi_int} ({filters[qi_int]}): non-matching GT ids {bad[:5]}"
            )
    print("[membership_check] all 50 sampled queries passed filtered GT predicate check", flush=True)


def write_native_hdf5(
    native_hdf5: Path,
    queries: np.ndarray,
    filters: Sequence[str],
    selectivities: np.ndarray,
    nres: np.ndarray,
    gt_ids: np.ndarray,
    gt_dists: np.ndarray,
    nn_ids: np.ndarray,
    nn_dists: np.ndarray,
    n_base: int = CASELAW_N_BASE,
) -> None:
    native_hdf5.parent.mkdir(parents=True, exist_ok=True)
    tmp = native_hdf5.with_suffix(".hdf5.part")
    if tmp.exists():
        tmp.unlink()
    nq, d = queries.shape
    dt_str = h5py.string_dtype(encoding="utf-8")
    with h5py.File(tmp, "w") as f:
        f.attrs["metric"] = "euclidean"
        f.attrs["dimension"] = int(d)
        f.attrs["N"] = int(n_base)
        f.attrs["k_gt"] = int(K_GT)
        f.attrs["source"] = json.dumps(source_urls())
        f.attrs["upstream_commit"] = UPSTREAM_COMMIT
        f.attrs["empty_gt_dropped_later"] = True

        f.create_dataset("test", data=queries.astype(np.float32, copy=False))
        f.create_dataset("filter", data=np.asarray(filters, dtype=object), dtype=dt_str)
        f.create_dataset("selectivity", data=selectivities.astype(np.float64, copy=False))
        f.create_dataset("nres", data=nres.astype(np.int32, copy=False))
        f.create_dataset("gt_ids", data=gt_ids.astype(np.int32, copy=False))
        f.create_dataset("gt_dists", data=gt_dists.astype(np.float32, copy=False))
        f.create_dataset("nn_ids", data=nn_ids.astype(np.int32, copy=False))
        f.create_dataset("nn_dists", data=nn_dists.astype(np.float32, copy=False))
        f.create_dataset("query_index", data=np.arange(nq, dtype=np.int32))
    tmp.replace(native_hdf5)
    print(f"[pack] wrote {native_hdf5} ({nq:,} queries, d={d})", flush=True)


def write_exact_gls_stats(
    stats_csv: Path,
    attrs: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
    nn_ids: np.ndarray,
) -> None:
    """Section 12: exact GLS at L=100 from unfiltered nn_ids on every population query."""
    t0 = time.time()
    records = manifest["queries"]
    rows = []
    for rec in records:
        qi = int(rec["query_index"])
        fstr = str(rec["filter"])
        sigma_g = float(rec["selectivity"])
        row_nn = nn_ids[qi]
        valid_nn = row_nn[row_nn >= 0]
        if valid_nn.size == 0:
            sigma_l = 0.0
        else:
            sub_attrs = {k: np.asarray(v)[valid_nn] for k, v in attrs.items()}
            mask = predicate_mask(sub_attrs, parse_predicate(fstr))
            sigma_l = float(mask.mean())
        denom = sigma_l + sigma_g
        rho = (sigma_l - sigma_g) / denom if denom > 0.0 else 0.0
        rows.append(
            {
                "q_id": f"qc{qi:05d}",
                "query_type": "cases",
                "filter": fstr,
                "selectivity": sigma_g,
                "correlation_ACORN": float(rho),
                "correlation_NEW": float(rho),
                "query_index": qi,
                "gls_L": int(K_GT),
            }
        )
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(stats_csv, index=False)
    print(
        f"[pack] wrote {stats_csv} ({len(rows):,} rows) in {time.time() - t0:.2f}s",
        flush=True,
    )


def build_pack(
    root: Path,
    qo_sample_frac: float = DEFAULT_QO_SAMPLE_FRAC,
    skip_download: bool = False,
    force_rebuild: bool = False,
) -> None:
    frac = validate_qo_sample_frac(qo_sample_frac)
    raw_dir = root / "raw"
    if not skip_download:
        download_raw_files(raw_dir)
    verify_raw_files_present(raw_dir)

    n_base, d_base = read_fbin_header(raw_dir / "caselaw_base_embeddings.bin")
    n_q, d_q = read_fbin_header(raw_dir / "caselaw_query_embeddings.bin")
    if n_base != CASELAW_N_BASE:
        raise RuntimeError(f"Base .fbin row count {n_base} != {CASELAW_N_BASE}")
    if n_q != CASELAW_N_QUERIES_OFFICIAL:
        raise RuntimeError(f"Query .fbin row count {n_q} != {CASELAW_N_QUERIES_OFFICIAL}")
    if d_base != d_q:
        raise RuntimeError(f"Base dim {d_base} != query dim {d_q}")

    base_hdf5 = root / "datasets" / "cases_dataset_0.hdf5"
    native_hdf5 = root / "native" / "cases_native.hdf5"
    stats_csv = root / "stats" / "filter_stats_0_k100.csv"
    manifest_path = root / "sample_manifest.json"
    metric_json = root / "metric_check.json"

    if force_rebuild or not _base_hdf5_valid(base_hdf5, d_base):
        base_meta = load_base_metadata(raw_dir)
        write_base_hdf5(
            base_hdf5,
            raw_dir / "caselaw_base_embeddings.bin",
            base_meta,
            n_base,
            d_base,
        )
        attrs = {
            "date_filed": base_meta["train_date_filed"],
            "court_jurisdiction": base_meta["train_court_jurisdiction"],
            "court_type": base_meta["train_court_type"],
            "court_full_name": base_meta["train_court_full_name"],
        }
    else:
        print(f"[pack] reusing verified {base_hdf5}", flush=True)
        attrs = load_attrs_from_base_hdf5(base_hdf5)

    filters = load_query_filters_and_check_metadata(raw_dir)
    queries = read_fbin(raw_dir / "caselaw_query_embeddings.bin")
    nn_ids, nn_dists = read_knn_gt(raw_dir / "caselaw_gt.bin")
    nres, gt_ids, gt_dists = read_range_gt(raw_dir / "caselaw_gt_filtered.bin", k_pad=K_GT)

    if nn_ids.shape != (CASELAW_N_QUERIES_OFFICIAL, K_GT):
        raise RuntimeError(f"Unfiltered GT shape {nn_ids.shape} != {(CASELAW_N_QUERIES_OFFICIAL, K_GT)}")
    if len(nres) != CASELAW_N_QUERIES_OFFICIAL:
        raise RuntimeError(f"Filtered GT nres length {len(nres)} != {CASELAW_N_QUERIES_OFFICIAL}")

    run_metric_check(base_hdf5, queries, nres, nn_ids, nn_dists, metric_json)
    run_membership_check(attrs, filters, nres, gt_ids, n_base=n_base)

    selectivities = compute_all_selectivities(attrs, filters, n_base=n_base)
    write_native_hdf5(
        native_hdf5,
        queries,
        filters,
        selectivities,
        nres,
        gt_ids,
        gt_dists,
        nn_ids,
        nn_dists,
        n_base=n_base,
    )

    manifest = build_sample_manifest(
        selectivities,
        nres,
        filters,
        frac=frac,
        seed=DEFAULT_SAMPLE_SEED,
        n_base=n_base,
    )
    manifest_path.write_bytes(dump_manifest_bytes(manifest))
    print(
        f"[pack] wrote {manifest_path}: n_population={manifest['n_population']:,}, "
        f"n_train={manifest['n_train']:,}, n_test={manifest['n_test']:,}, "
        f"n_dropped_empty_gt={manifest['n_dropped_empty_gt']}",
        flush=True,
    )

    write_exact_gls_stats(stats_csv, attrs, manifest, nn_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qo-sample-frac",
        type=float,
        default=DEFAULT_QO_SAMPLE_FRAC,
        help="Stratified training sample fraction in [0.05, 0.20] (default: 0.10).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Rebuild pack from existing raw/ files without downloading.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rewriting cases_dataset_0.hdf5 even if already valid.",
    )
    parser.add_argument(
        "--pack-root",
        type=Path,
        default=None,
        help="Override CaseLaw_7M pack root directory.",
    )
    args = parser.parse_args()

    validate_qo_sample_frac(args.qo_sample_frac)
    root = args.pack_root if args.pack_root is not None else (REPO_ROOT / pack_root("caselaw"))
    build_pack(
        root,
        qo_sample_frac=args.qo_sample_frac,
        skip_download=args.skip_download,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
