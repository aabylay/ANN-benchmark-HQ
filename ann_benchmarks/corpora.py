"""Corpus registry and CaseLaw sampling/recall helpers for MoReVec and CaseLaw."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

CORPORA = ("morevec", "caselaw")

CASELAW_SOURCE_CASES = 8_362_175
CASELAW_N_BASE = 7_414_023
CASELAW_N_QUERIES_OFFICIAL = 89_812

MIN_QO_SAMPLE_FRAC = 0.05
MAX_QO_SAMPLE_FRAC = 0.20
DEFAULT_QO_SAMPLE_FRAC = 0.10
DEFAULT_SAMPLE_SEED = 42

BIN_NAMES: Tuple[str, ...] = ("low", "medium", "high", "residual")

_TABLES: Dict[str, Tuple[str, ...]] = {
    "morevec": ("movies", "reviews"),
    "caselaw": ("cases",),
}

_CARDINALITIES: Dict[Tuple[str, str], int] = {
    ("morevec", "movies"): 551_155,
    ("morevec", "reviews"): 2_598_267,
    ("caselaw", "cases"): CASELAW_N_BASE,
}

_JSON_OP_MAP: Dict[str, str] = {
    "$eq": "=",
    "$ne": "!=",
    "$gte": ">=",
    "$lte": "<=",
    "$gt": ">",
    "$lt": "<",
}


def validate_corpus(corpus: str) -> str:
    if corpus not in CORPORA:
        raise ValueError(f"Unknown corpus {corpus!r}; expected one of {CORPORA}")
    return corpus


def pack_root(corpus: str, dataset_size: str = "large") -> str:
    validate_corpus(corpus)
    if corpus == "caselaw":
        return os.path.join("data", "datasets", "CaseLaw_7M")
    return os.path.join("data", "datasets", f"MoRe_{dataset_size}")


def tables_for(corpus: str) -> List[str]:
    validate_corpus(corpus)
    return list(_TABLES[corpus])


def result_dir(
    corpus: str,
    workload: str = "flex",
    dataset_size: str = "large",
    data_table: str = "movies",
    att_idx: int = 0,
) -> str:
    validate_corpus(corpus)
    if corpus == "caselaw" or workload == "native":
        table = data_table if corpus == "caselaw" else "cases"
        return os.path.join("results", f"CaseLaw_7M_native_{table}")
    if workload in ("hard", "superhard"):
        return os.path.join("results", f"MoRe_UPD_{dataset_size}_{workload}_{data_table}")
    if workload == "flex":
        return os.path.join("results", f"MoRe_UPD_{dataset_size}_attidx_{att_idx}")
    raise ValueError(f"Unknown workload {workload!r} for corpus {corpus!r}")


def cardinality(corpus: str, table: str) -> int:
    validate_corpus(corpus)
    key = (corpus, table)
    if key not in _CARDINALITIES:
        raise KeyError(f"Unknown (corpus, table) pair: {key!r}")
    return _CARDINALITIES[key]


def validate_qo_sample_frac(frac: float) -> float:
    f = float(frac)
    if not math.isfinite(f) or f < MIN_QO_SAMPLE_FRAC or f > MAX_QO_SAMPLE_FRAC:
        raise ValueError(
            f"qo_sample_frac must be in [{MIN_QO_SAMPLE_FRAC}, {MAX_QO_SAMPLE_FRAC}], got {frac!r}"
        )
    return f


def selectivity_bin(sigma: float) -> str:
    s = float(sigma)
    if 0.005 <= s < 0.01:
        return "low"
    if 0.01 <= s < 0.1:
        return "medium"
    if 0.1 <= s < 0.5:
        return "high"
    return "residual"


def allocate_bin_quotas(
    bin_counts: Mapping[str, int],
    frac: float,
) -> Dict[str, int]:
    f = validate_qo_sample_frac(frac)
    total_pop = sum(int(bin_counts.get(b, 0)) for b in BIN_NAMES)
    target = int(round(f * total_pop))
    raw = {b: f * int(bin_counts.get(b, 0)) for b in BIN_NAMES}
    floors = {b: int(math.floor(raw[b])) for b in BIN_NAMES}
    extra = target - sum(floors.values())
    # Sort descending by remainder (raw - floor), ties broken by bin name ascending.
    ordered_bins = sorted(
        BIN_NAMES,
        key=lambda b: (-(raw[b] - floors[b]), b),
    )
    quotas = dict(floors)
    for b in ordered_bins[: max(0, extra)]:
        quotas[b] += 1
    return quotas


def draw_stratified_sample(
    selectivities: Sequence[float],
    nres: Sequence[int] | None = None,
    frac: float = DEFAULT_QO_SAMPLE_FRAC,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Return (train_indices_sorted, bin_counts_population, bin_counts_train)."""
    f = validate_qo_sample_frac(frac)
    sels = np.asarray(selectivities, dtype=np.float64)
    if nres is None:
        nres_arr = np.ones(len(sels), dtype=np.int32)
    else:
        nres_arr = np.asarray(nres, dtype=np.int32)
    if len(sels) != len(nres_arr):
        raise ValueError("selectivities and nres must have the same length")

    by_bin: Dict[str, List[int]] = {b: [] for b in BIN_NAMES}
    for idx in range(len(sels)):
        if int(nres_arr[idx]) <= 0:
            continue
        b = selectivity_bin(float(sels[idx]))
        by_bin[b].append(idx)

    bin_counts_pop = {b: len(by_bin[b]) for b in BIN_NAMES}
    quotas = allocate_bin_quotas(bin_counts_pop, f)
    selected: List[int] = []
    for b in BIN_NAMES:
        indices = np.asarray(sorted(by_bin[b]), dtype=np.int64)
        q = quotas[b]
        if q <= 0:
            continue
        chosen = np.random.RandomState(int(seed)).choice(indices, size=q, replace=False)
        selected.extend(int(x) for x in chosen)
    selected.sort()
    return np.asarray(selected, dtype=np.int64), bin_counts_pop, quotas


def build_sample_manifest(
    selectivities: Sequence[float],
    nres: Sequence[int],
    filters: Sequence[str],
    frac: float = DEFAULT_QO_SAMPLE_FRAC,
    seed: int = DEFAULT_SAMPLE_SEED,
    n_base: int = CASELAW_N_BASE,
) -> Dict[str, Any]:
    f = validate_qo_sample_frac(frac)
    sels = np.asarray(selectivities, dtype=np.float64)
    nres_arr = np.asarray(nres, dtype=np.int32)
    if not (len(sels) == len(nres_arr) == len(filters)):
        raise ValueError("selectivities, nres, and filters must have the same length")

    train_idx, bin_counts_pop, bin_counts_train = draw_stratified_sample(
        sels, nres_arr, frac=f, seed=seed
    )
    train_set = set(int(x) for x in train_idx.tolist())
    n_official = len(sels)
    n_dropped = int(np.sum(nres_arr == 0))
    queries: List[Dict[str, Any]] = []
    for qi in range(n_official):
        nr = int(nres_arr[qi])
        if nr == 0:
            continue
        sel = float(sels[qi])
        b = selectivity_bin(sel)
        flt = filters[qi]
        if isinstance(flt, (bytes, np.bytes_)):
            flt = flt.decode("utf-8")
        queries.append(
            {
                "query_index": int(qi),
                "selectivity": sel,
                "bin": b,
                "split": "train" if qi in train_set else "test",
                "nres": nr,
                "filter": str(flt),
            }
        )
    n_pop = len(queries)
    n_train = len(train_set)
    n_test = n_pop - n_train
    return {
        "corpus": "caselaw",
        "frac": f,
        "seed": int(seed),
        "N": int(n_base),
        "n_queries_official": int(n_official),
        "n_dropped_empty_gt": int(n_dropped),
        "n_population": int(n_pop),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_sample": int(n_train),
        "bin_counts_population": {b: int(bin_counts_pop[b]) for b in BIN_NAMES},
        "bin_counts_train": {b: int(bin_counts_train[b]) for b in BIN_NAMES},
        "bin_counts_sample": {b: int(bin_counts_train[b]) for b in BIN_NAMES},
        "queries": queries,
    }


def dump_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    text = json.dumps(manifest, sort_keys=True, separators=(", ", ": ")) + "\n"
    return text.encode("utf-8")


def canonical_filter_from_json(filter_obj: Mapping[str, Any]) -> str:
    """Convert one upstream filter dict from caselaw_query_filters.jsonl into canonical string."""
    if "$and" in filter_obj:
        items = list(filter_obj["$and"])
    else:
        items = [filter_obj]
    if not items:
        raise ValueError(f"Empty filter object: {filter_obj!r}")
    clauses: List[str] = []
    for item in items:
        if not isinstance(item, Mapping) or len(item) != 1:
            raise ValueError(f"Expected single-field clause dict, got {item!r}")
        field, cond = next(iter(item.items()))
        if not isinstance(cond, Mapping) or len(cond) != 1:
            raise ValueError(f"Expected single-op dict for {field!r}, got {cond!r}")
        raw_op, val = next(iter(cond.items()))
        if raw_op not in _JSON_OP_MAP:
            raise ValueError(f"Unsupported JSON operator {raw_op!r} in {filter_obj!r}")
        op = _JSON_OP_MAP[raw_op]
        if field == "date_filed":
            clauses.append(f"{field} {op} {int(val)}")
        elif field in ("court_jurisdiction", "court_type", "court_full_name"):
            sval = str(val)
            if '"' in sval:
                raise ValueError(f"Quoted string value contains double quote: {sval!r}")
            clauses.append(f'{field} {op} "{sval}"')
        else:
            raise ValueError(f"Unknown CaseLaw filter field {field!r}")
    return " AND ".join(clauses)


def caselaw_recall_at_k(
    returned_ids: Sequence[int],
    gt_row: Sequence[int],
    nres: int,
    k: int = 10,
) -> float:
    """Recall@k per section 10.5: denom = min(k, nres), G = gt_ids[:denom] without -1."""
    nr = int(nres)
    if nr <= 0:
        raise ValueError("nres == 0 has undefined recall denominator and must be dropped")
    denom = min(int(k), nr)
    g_prefix = [int(x) for x in gt_row[:denom] if int(x) != -1]
    pred_prefix = [int(x) for x in returned_ids[: int(k)] if int(x) != -1]
    hits = len(set(pred_prefix) & set(g_prefix))
    return float(hits) / float(denom)
