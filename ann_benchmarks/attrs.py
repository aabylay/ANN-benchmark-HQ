"""Attribute name → column resolution for multi-attr filtered ANN.

Shared by flex / hard / superhard / native workload modes. Maps filter strings
such as ``avg_rating >= 7.0`` or ``year <= 2015`` or CaseLaw conjunctions onto
the columns loaded from train HDF5s.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy

# Column order matches load_train_dataset stacking in runner.py.
MOVIES_ATTR_NAMES = ["avgrating", "title", "genre", "num_votes", "year"]
REVIEWS_ATTR_NAMES = ["mid", "uid", "likeshare", "movierating", "total_votes"]
CASES_ATTR_NAMES = [
    "date_filed",
    "court_jurisdiction",
    "court_type",
    "court_full_name",
]

# Numeric columns that may appear in filter predicates (SQL + FAISS masks).
MOVIES_NUMERIC_ATTRS = ["avgrating", "num_votes", "year"]
REVIEWS_NUMERIC_ATTRS = ["likeshare", "movierating", "total_votes"]
CASES_NUMERIC_ATTRS = ["date_filed"]
CASES_CATEGORICAL_ATTRS = [
    "court_jurisdiction",
    "court_type",
    "court_full_name",
]

PRIMARY_ATTR = {
    "movies": "avgrating",
    "reviews": "total_votes",
    "cases": "date_filed",
}

# Filter-string spellings → canonical column names used in attrs dict / SQL.
ATTR_ALIASES = {
    "avg_rating": "avgrating",
    "avgrating": "avgrating",
    "num_votes": "num_votes",
    "numvotes": "num_votes",
    "year": "year",
    "total_votes": "total_votes",
    "totalvotes": "total_votes",
    "movierating": "movierating",
    "movie_rating": "movierating",
    "likeshare": "likeshare",
    "like_share": "likeshare",
    "mid": "mid",
    "uid": "uid",
    "title": "title",
    "genre": "genre",
    "date_filed": "date_filed",
    "court_jurisdiction": "court_jurisdiction",
    "court_type": "court_type",
    "court_full_name": "court_full_name",
}

ALLOWED_OPS = frozenset({">=", "<=", ">", "<", "=", "!="})
_CLAUSE_OPS: Tuple[str, ...] = (">=", "<=", "!=", "=", ">", "<")
_AND_SEP = " AND "

AttrsDict = Dict[str, numpy.ndarray]
ClauseTriple = Tuple[str, str, str]


def attr_names_for(dataset_type: str) -> list[str]:
    if dataset_type == "movies":
        return list(MOVIES_ATTR_NAMES)
    if dataset_type == "reviews":
        return list(REVIEWS_ATTR_NAMES)
    if dataset_type == "cases":
        return list(CASES_ATTR_NAMES)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def numeric_attr_names_for(dataset_type: str) -> list[str]:
    if dataset_type == "movies":
        return list(MOVIES_NUMERIC_ATTRS)
    if dataset_type == "reviews":
        return list(REVIEWS_NUMERIC_ATTRS)
    if dataset_type == "cases":
        return list(CASES_NUMERIC_ATTRS)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def categorical_attr_names_for(dataset_type: str) -> list[str]:
    if dataset_type == "movies":
        return ["title", "genre"]
    if dataset_type == "reviews":
        return ["mid", "uid"]
    if dataset_type == "cases":
        return list(CASES_CATEGORICAL_ATTRS)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def build_attrs_dict(
    X_attrs: Union[Mapping[str, numpy.ndarray], numpy.ndarray],
    dataset_type: str,
) -> AttrsDict:
    """Build name→array map from stacked train attributes (shape 5 x N) or dict."""
    if isinstance(X_attrs, Mapping):
        return {k: numpy.asarray(v) for k, v in X_attrs.items()}
    names = attr_names_for(dataset_type)
    if X_attrs.shape[0] != len(names):
        raise ValueError(
            f"Expected {len(names)} attribute rows for {dataset_type}, "
            f"got {X_attrs.shape[0]}"
        )
    out: AttrsDict = {}
    for i, name in enumerate(names):
        col = numpy.asarray(X_attrs[i])
        if name in numeric_attr_names_for(dataset_type):
            # Keep IEEE float64 from the HDF5. Casting to float32 and then
            # storing as PostgreSQL FLOAT8 rewrites exact tenths such as 9.2
            # to 9.199999809265137, so ``avgrating >= 9.2`` drops those rows
            # and pgvector brute-force recall falls below 1.0.
            out[name] = col.astype(numpy.float64, copy=False)
        else:
            out[name] = col
    return out


def canonicalize_attr_name(attr: str) -> str:
    key = attr.strip()
    if key in ATTR_ALIASES:
        return ATTR_ALIASES[key]
    norm = key.replace("_", "").lower()
    for alias, canon in ATTR_ALIASES.items():
        if alias.replace("_", "").lower() == norm:
            return canon
    raise KeyError(f"Unknown filter attribute: {attr!r}")


def resolve_attr_array(attrs: Mapping[str, numpy.ndarray], attr: str) -> numpy.ndarray:
    canon = canonicalize_attr_name(attr)
    if canon not in attrs:
        raise KeyError(
            f"Attribute {attr!r} (canonical {canon!r}) not in attrs "
            f"(have {sorted(attrs)})"
        )
    return numpy.asarray(attrs[canon])


def sql_column_for_attr(attr: str, dataset_type: str) -> str:
    """Whitelist-resolved SQL column name for a filter attribute."""
    canon = canonicalize_attr_name(attr)
    allowed = set(numeric_attr_names_for(dataset_type))
    if canon not in allowed:
        raise KeyError(
            f"Attribute {attr!r} is not a numeric filter column for "
            f"{dataset_type} (allowed: {sorted(allowed)})"
        )
    return canon


def coerce_attrs(
    X_attr: Union[Mapping[str, numpy.ndarray], numpy.ndarray],
    dataset_type: str,
) -> AttrsDict:
    """Accept attrs dict or legacy single primary column array."""
    if isinstance(X_attr, Mapping):
        return {k: numpy.asarray(v) for k, v in X_attr.items()}
    primary = PRIMARY_ATTR[dataset_type]
    return {primary: numpy.asarray(X_attr, dtype=numpy.float32)}


def primary_attr_array(attrs: AttrsDict, dataset_type: str) -> numpy.ndarray:
    return numpy.asarray(attrs[PRIMARY_ATTR[dataset_type]], dtype=numpy.float32)


def validate_op(op: str) -> str:
    if op not in ALLOWED_OPS:
        raise ValueError(f"Unsupported filter operator: {op}")
    return op


def _is_clause_list(fvalue) -> bool:
    """Return True if fvalue is a sequence of (attr, op, value) triples."""
    if not isinstance(fvalue, (list, tuple)) or not fvalue:
        return False
    first = fvalue[0]
    return (
        isinstance(first, (list, tuple))
        and not isinstance(first, str)
        and len(first) == 3
    )


def _split_and(text: str) -> List[str]:
    """Split on the four characters space-A-N-D-space outside double quotes."""
    parts: List[str] = []
    buf: List[str] = []
    in_quote = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
            i += 1
            continue
        if not in_quote and text.startswith(_AND_SEP, i):
            parts.append("".join(buf))
            buf = []
            i += len(_AND_SEP)
            continue
        buf.append(ch)
        i += 1
    if in_quote:
        raise ValueError(f"unterminated quote in predicate: {text!r}")
    parts.append("".join(buf))
    return parts


def _unquote_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("empty filter value")
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        inner = value[1:-1]
        if '"' in inner:
            raise ValueError(f"quoted filter value contains a quote: {raw!r}")
        return inner
    if '"' in value:
        raise ValueError(f"malformed quoted filter value: {raw!r}")
    return value


def _parse_clause(clause: str) -> ClauseTriple:
    text = clause.strip()
    if not text:
        raise ValueError("empty filter clause")
    in_quote = False
    found = None
    for i, ch in enumerate(text):
        if ch == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        for op in _CLAUSE_OPS:
            if text.startswith(op, i):
                found = (i, op)
                break
        if found is not None:
            break
    if found is None:
        raise ValueError(f"cannot parse filter clause: {clause!r}")
    idx, op = found
    attr = text[:idx].strip()
    raw_val = text[idx + len(op) :]
    if not attr:
        raise ValueError(f"filter clause is missing an attribute: {clause!r}")
    value = _unquote_value(raw_val)
    validate_op(op)
    return (attr, op, value)


def parse_predicate(s: Union[str, Sequence]) -> List:
    """Parse a canonical predicate string into a list of (attr, op, value) triples.

    - ``No_filter`` or ``["No_filter"]`` returns ``["No_filter"]``.
    - One clause returns ``[(attr, op, value)]``.
    - Conjunctions separated by `` AND `` return one triple per clause.
    - Quoted string values preserve internal spaces and strip outer quotes.
    """
    if s == ["No_filter"] or s == "No_filter":
        return ["No_filter"]
    if _is_clause_list(s):
        return [(str(c[0]), validate_op(str(c[1])), str(c[2])) for c in s]
    if not isinstance(s, str):
        raise TypeError(f"predicate must be a string, got {type(s).__name__}")
    stripped = s.strip()
    if stripped == "No_filter":
        return ["No_filter"]
    clauses = _split_and(s)
    if any(not part.strip() for part in clauses):
        raise ValueError(f"empty clause in predicate: {s!r}")
    return [_parse_clause(part) for part in clauses]


_TEXT_ARRAY_CACHE: Dict[int, Tuple[numpy.ndarray, numpy.ndarray]] = {}
_CAT_INDEX_CACHE: Dict[int, Tuple[numpy.ndarray, Dict[str, Tuple[int, int]], numpy.ndarray]] = {}
_CAT_PAIR_INDEX_CACHE: Dict[
    Tuple[int, int],
    Tuple[numpy.ndarray, numpy.ndarray, Dict[Tuple[str, str], Tuple[int, int]], numpy.ndarray],
] = {}
_CAT_DATE_INDEX_CACHE: Dict[
    Tuple[int, int],
    Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Tuple[int, int]], numpy.ndarray, numpy.ndarray],
] = {}
_DATE_SORT_CACHE: Dict[int, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]] = {}


def _factorize_strings(col: numpy.ndarray) -> Tuple[List[str], numpy.ndarray]:
    """Fast O(N) string-to-code mapping without O(N log N) Python str comparisons."""
    lut: Dict[str, int] = {}
    uniques: List[str] = []
    codes = numpy.empty(len(col), dtype=numpy.int32)
    get = lut.get
    for i, val in enumerate(col):
        s = val if isinstance(val, str) else str(val)
        c = get(s)
        if c is None:
            c = len(uniques)
            lut[s] = c
            uniques.append(s)
        codes[i] = c
    return uniques, codes


def _as_text_array(arr: numpy.ndarray) -> numpy.ndarray:
    values = numpy.asarray(arr)
    if values.dtype.kind == "U":
        return values
    if values.dtype.kind == "O":
        if values.size == 0 or isinstance(values.flat[0], str):
            return values
    key = id(arr)
    cached = _TEXT_ARRAY_CACHE.get(key)
    if cached is not None and cached[0] is arr:
        return cached[1]
    decoded = numpy.array(
        [
            x.decode("utf-8") if isinstance(x, (bytes, numpy.bytes_)) else str(x)
            for x in values
        ],
        dtype=object,
    )
    if values.size >= 1024:
        _TEXT_ARRAY_CACHE[key] = (arr, decoded)
    return decoded


def _get_cat_date_index(cat_arr: numpy.ndarray, date_arr: numpy.ndarray):
    key = (id(cat_arr), id(date_arr))
    cached = _CAT_DATE_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is cat_arr and cached[1] is date_arr:
        return cached[2], cached[3], cached[4]
    col = _as_text_array(cat_arr)
    dates = numpy.asarray(date_arr, dtype=numpy.int64)
    cats, codes = _factorize_strings(col)
    order = numpy.lexsort((dates, codes))
    s_codes = codes[order]
    s_dates = dates[order]
    s_rows = order.astype(numpy.int32)
    cuts = numpy.searchsorted(s_codes, numpy.arange(len(cats) + 1))
    bounds = {str(c): (int(cuts[i]), int(cuts[i + 1])) for i, c in enumerate(cats)}
    _CAT_DATE_INDEX_CACHE[key] = (cat_arr, date_arr, bounds, s_dates, s_rows)
    return bounds, s_dates, s_rows


def _get_cat_index(cat_arr: numpy.ndarray):
    key = id(cat_arr)
    cached = _CAT_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is cat_arr:
        return cached[1], cached[2]
    col = _as_text_array(cat_arr)
    cats, codes = _factorize_strings(col)
    order = numpy.argsort(codes, kind="stable")
    s_codes = codes[order]
    s_rows = order.astype(numpy.int32)
    cuts = numpy.searchsorted(s_codes, numpy.arange(len(cats) + 1))
    bounds = {str(c): (int(cuts[i]), int(cuts[i + 1])) for i, c in enumerate(cats)}
    _CAT_INDEX_CACHE[key] = (cat_arr, bounds, s_rows)
    return bounds, s_rows


def _get_cat_pair_index(arr1: numpy.ndarray, arr2: numpy.ndarray):
    key = (id(arr1), id(arr2))
    cached = _CAT_PAIR_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is arr1 and cached[1] is arr2:
        return cached[2], cached[3]
    col1 = _as_text_array(arr1)
    col2 = _as_text_array(arr2)
    cats1, codes1 = _factorize_strings(col1)
    cats2, codes2 = _factorize_strings(col2)
    n2 = max(len(cats2), 1)
    pair_codes = codes1.astype(numpy.int64) * n2 + codes2.astype(numpy.int64)
    order = numpy.argsort(pair_codes, kind="stable")
    s_pair = pair_codes[order]
    s_rows = order.astype(numpy.int32)
    uniq_pairs, first_idx = numpy.unique(s_pair, return_index=True)
    ends = numpy.append(first_idx[1:], len(s_pair))
    bounds: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for pcode, start, end in zip(uniq_pairs, first_idx, ends):
        c1_idx = int(pcode // n2)
        c2_idx = int(pcode % n2)
        bounds[(cats1[c1_idx], cats2[c2_idx])] = (int(start), int(end))
    _CAT_PAIR_INDEX_CACHE[key] = (arr1, arr2, bounds, s_rows)
    return bounds, s_rows


def _get_date_index(date_arr: numpy.ndarray):
    key = id(date_arr)
    cached = _DATE_SORT_CACHE.get(key)
    if cached is not None and cached[0] is date_arr:
        return cached[1], cached[2]
    dates = numpy.asarray(date_arr, dtype=numpy.int64)
    order = numpy.argsort(dates, kind="stable")
    s_dates = dates[order]
    s_rows = order.astype(numpy.int32)
    _DATE_SORT_CACHE[key] = (date_arr, s_dates, s_rows)
    return s_dates, s_rows


def _try_fast_caselaw_mask(
    attrs: Mapping[str, numpy.ndarray],
    clauses: Sequence[ClauseTriple],
    n: int,
) -> Union[numpy.ndarray, None]:
    """Fast CSR path for large CaseLaw tables (bit-for-bit identical to _clause_mask)."""
    if n < 10_000 or "date_filed" not in attrs:
        return None
    cat_eq: List[Tuple[str, str]] = []
    date_ge: List[int] = []
    date_le: List[int] = []
    for raw_attr, op, val in clauses:
        canon = canonicalize_attr_name(raw_attr)
        if canon == "date_filed":
            try:
                ival = int(val)
            except ValueError:
                return None
            if op == ">=":
                date_ge.append(ival)
            elif op == ">":
                date_ge.append(ival + 1)
            elif op == "<=":
                date_le.append(ival)
            elif op == "<":
                date_le.append(ival - 1)
            elif op == "=":
                date_ge.append(ival)
                date_le.append(ival)
            else:
                return None
        elif canon in CASES_CATEGORICAL_ATTRS and op == "=":
            cat_eq.append((canon, str(val)))
        else:
            return None

    lo_val = max(date_ge) if date_ge else None
    hi_val = min(date_le) if date_le else None
    if lo_val is not None and hi_val is not None and lo_val > hi_val:
        return numpy.zeros(n, dtype=bool)

    date_arr = attrs["date_filed"]
    if len(cat_eq) == 1 and (lo_val is not None or hi_val is not None):
        cat_name, cat_val = cat_eq[0]
        bounds, s_dates, s_rows = _get_cat_date_index(attrs[cat_name], date_arr)
        span = bounds.get(cat_val)
        if span is None:
            return numpy.zeros(n, dtype=bool)
        c_lo, c_hi = span
        d_slice = s_dates[c_lo:c_hi]
        r_slice = s_rows[c_lo:c_hi]
        i_lo = int(numpy.searchsorted(d_slice, lo_val, side="left")) if lo_val is not None else 0
        i_hi = int(numpy.searchsorted(d_slice, hi_val, side="right")) if hi_val is not None else len(d_slice)
        mask = numpy.zeros(n, dtype=bool)
        if i_hi > i_lo:
            mask[r_slice[i_lo:i_hi]] = True
        return mask

    if len(cat_eq) == 1 and lo_val is None and hi_val is None:
        cat_name, cat_val = cat_eq[0]
        bounds, s_rows = _get_cat_index(attrs[cat_name])
        span = bounds.get(cat_val)
        if span is None:
            return numpy.zeros(n, dtype=bool)
        c_lo, c_hi = span
        mask = numpy.zeros(n, dtype=bool)
        if c_hi > c_lo:
            mask[s_rows[c_lo:c_hi]] = True
        return mask

    if len(cat_eq) == 2 and lo_val is None and hi_val is None:
        (c1, v1), (c2, v2) = cat_eq
        bounds, s_rows = _get_cat_pair_index(attrs[c1], attrs[c2])
        span = bounds.get((v1, v2))
        mask = numpy.zeros(n, dtype=bool)
        if span is not None and span[1] > span[0]:
            mask[s_rows[span[0] : span[1]]] = True
        return mask

    if len(cat_eq) == 0 and (lo_val is not None or hi_val is not None):
        s_dates, s_rows = _get_date_index(date_arr)
        i_lo = int(numpy.searchsorted(s_dates, lo_val, side="left")) if lo_val is not None else 0
        i_hi = int(numpy.searchsorted(s_dates, hi_val, side="right")) if hi_val is not None else len(s_dates)
        if (i_hi - i_lo) <= (n // 3):
            mask = numpy.zeros(n, dtype=bool)
            if i_hi > i_lo:
                mask[s_rows[i_lo:i_hi]] = True
            return mask

    return None


def _clause_mask(attrs: Mapping[str, numpy.ndarray], clause: Sequence[str]) -> numpy.ndarray:
    """Evaluate a single (attr, op, value) triple against attrs."""
    from ann_benchmarks.algorithms.faiss.postfilter import filter_mask_from_fvalue

    attr, op, value = clause[0], clause[1], clause[2]
    validate_op(op)
    arr = resolve_attr_array(attrs, attr)
    if numpy.issubdtype(arr.dtype, numpy.number):
        if numpy.issubdtype(arr.dtype, numpy.integer):
            try:
                threshold = int(value)
            except ValueError:
                threshold = float(value)
            if op == ">=":
                return arr >= threshold
            if op == "<=":
                return arr <= threshold
            if op == ">":
                return arr > threshold
            if op == "<":
                return arr < threshold
            if op == "=":
                return arr == threshold
            if op == "!=":
                return arr != threshold
        return filter_mask_from_fvalue(arr.astype(numpy.float32, copy=False), [attr, op, value])

    col = _as_text_array(arr)
    text = str(value)
    if op == "=":
        return col == text
    if op == "!=":
        return col != text
    if op == ">=":
        return col >= text
    if op == "<=":
        return col <= text
    if op == ">":
        return col > text
    if op == "<":
        return col < text
    raise ValueError(f"Unsupported filter operator: {op}")


def predicate_mask(attrs: Mapping[str, numpy.ndarray], predicate) -> numpy.ndarray:
    """Boolean mask for ``No_filter``, one triple, or a conjunction of triples.

    - ``["No_filter"]`` or ``"No_filter"`` returns an all-true mask.
    - A single numeric triple calls ``filter_mask_from_fvalue`` so MoRe numeric
      masks stay on that function.
    - A string value compares the decoded string column without calling ``float()``.
    - Multiple clauses return the bitwise AND of the individual clause masks.
    """
    from ann_benchmarks.algorithms.faiss.postfilter import filter_mask_from_fvalue

    n = len(next(iter(attrs.values())))
    if predicate == ["No_filter"] or predicate == "No_filter":
        return numpy.ones(n, dtype=bool)
    if isinstance(predicate, str):
        predicate = parse_predicate(predicate)
        if predicate == ["No_filter"]:
            return numpy.ones(n, dtype=bool)

    if _is_clause_list(predicate):
        clauses = [(str(c[0]), str(c[1]), str(c[2])) for c in predicate]
    elif (
        isinstance(predicate, (list, tuple))
        and len(predicate) == 3
        and isinstance(predicate[0], str)
    ):
        clauses = [(str(predicate[0]), str(predicate[1]), str(predicate[2]))]
    else:
        raise ValueError(f"unsupported predicate: {predicate!r}")

    if len(clauses) == 1:
        attr, op, val = clauses[0]
        arr = resolve_attr_array(attrs, attr)
        if numpy.issubdtype(arr.dtype, numpy.floating):
            return filter_mask_from_fvalue(
                arr.astype(numpy.float32, copy=False), [attr, op, val]
            )
        if numpy.issubdtype(arr.dtype, numpy.integer) and n < 10_000:
            return filter_mask_from_fvalue(arr, [attr, op, val])

    fast = _try_fast_caselaw_mask(attrs, clauses, n)
    if fast is not None:
        return fast

    if len(clauses) == 1:
        attr, op, val = clauses[0]
        arr = resolve_attr_array(attrs, attr)
        if numpy.issubdtype(arr.dtype, numpy.number):
            col = arr if numpy.issubdtype(arr.dtype, numpy.integer) else arr.astype(numpy.float32, copy=False)
            return filter_mask_from_fvalue(col, [attr, op, val])

    mask = numpy.ones(n, dtype=bool)
    for clause in clauses:
        mask &= numpy.asarray(_clause_mask(attrs, clause), dtype=bool)
    return mask


def prepare_caselaw_attr_indices(attrs: Mapping[str, numpy.ndarray]) -> None:
    """Pre-warm CSR attribute indices for CaseLaw before query timing starts."""
    if "date_filed" not in attrs:
        return
    date_arr = attrs["date_filed"]
    if len(date_arr) < 10_000:
        return
    _get_date_index(date_arr)
    for cat in CASES_CATEGORICAL_ATTRS:
        if cat in attrs:
            _get_cat_index(attrs[cat])
            _get_cat_date_index(attrs[cat], date_arr)
    if "court_type" in attrs and "court_jurisdiction" in attrs:
        _get_cat_pair_index(attrs["court_type"], attrs["court_jurisdiction"])
        _get_cat_pair_index(attrs["court_jurisdiction"], attrs["court_type"])


def _count_caselaw_clauses(
    attrs: Mapping[str, numpy.ndarray],
    clauses: Sequence[ClauseTriple],
    n: int,
) -> Union[int, None]:
    if n < 10_000 or "date_filed" not in attrs:
        return None
    cat_eq: List[Tuple[str, str]] = []
    date_ge: List[int] = []
    date_le: List[int] = []
    for raw_attr, op, val in clauses:
        canon = canonicalize_attr_name(raw_attr)
        if canon == "date_filed":
            try:
                ival = int(val)
            except ValueError:
                return None
            if op == ">=":
                date_ge.append(ival)
            elif op == ">":
                date_ge.append(ival + 1)
            elif op == "<=":
                date_le.append(ival)
            elif op == "<":
                date_le.append(ival - 1)
            elif op == "=":
                date_ge.append(ival)
                date_le.append(ival)
            else:
                return None
        elif canon in CASES_CATEGORICAL_ATTRS and op == "=":
            cat_eq.append((canon, str(val)))
        else:
            return None

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
        i_lo = int(numpy.searchsorted(d_slice, lo_val, side="left")) if lo_val is not None else 0
        i_hi = int(numpy.searchsorted(d_slice, hi_val, side="right")) if hi_val is not None else len(d_slice)
        return max(0, i_hi - i_lo)

    if len(cat_eq) == 1 and lo_val is None and hi_val is None:
        cat_name, cat_val = cat_eq[0]
        bounds, _ = _get_cat_index(attrs[cat_name])
        span = bounds.get(cat_val)
        return 0 if span is None else max(0, span[1] - span[0])

    if len(cat_eq) == 2 and lo_val is None and hi_val is None:
        (c1, v1), (c2, v2) = cat_eq
        bounds, _ = _get_cat_pair_index(attrs[c1], attrs[c2])
        span = bounds.get((v1, v2))
        return 0 if span is None else max(0, span[1] - span[0])

    if len(cat_eq) == 0 and (lo_val is not None or hi_val is not None):
        s_dates, _ = _get_date_index(date_arr)
        i_lo = int(numpy.searchsorted(s_dates, lo_val, side="left")) if lo_val is not None else 0
        i_hi = int(numpy.searchsorted(s_dates, hi_val, side="right")) if hi_val is not None else len(s_dates)
        return max(0, i_hi - i_lo)

    return None


def selectivity_from_attrs(attrs: Mapping[str, numpy.ndarray], fvalue) -> float:
    """Exact global selectivity of fvalue over attrs."""
    n = len(next(iter(attrs.values())))
    if n == 0:
        return 0.0
    if fvalue == ["No_filter"] or fvalue == "No_filter":
        return 1.0
    if _is_clause_list(fvalue):
        clauses = [(str(c[0]), str(c[1]), str(c[2])) for c in fvalue]
        cnt = _count_caselaw_clauses(attrs, clauses, n)
        if cnt is not None:
            return float(cnt) / float(n)
    return float(filter_mask_from_attrs(attrs, fvalue).mean())


def postfilter_candidates_from_attrs(
    attrs: Mapping[str, numpy.ndarray],
    fvalue,
    indices: numpy.ndarray,
    target_k: int,
) -> numpy.ndarray:
    """Keep the first target_k candidate indices (in search order) that pass fvalue."""
    idx_arr = numpy.asarray(indices, dtype=numpy.int64)
    valid = idx_arr[idx_arr >= 0]
    if valid.size == 0:
        return numpy.empty(0, dtype=numpy.int64)
    if fvalue == ["No_filter"] or fvalue == "No_filter":
        return valid[: int(target_k)].copy()
    sub_attrs = {k: numpy.asarray(v)[valid] for k, v in attrs.items()}
    if _is_clause_list(fvalue):
        pass_mask = predicate_mask(sub_attrs, fvalue)
    else:
        pass_mask = filter_mask_from_attrs(sub_attrs, fvalue)
    return valid[pass_mask][: int(target_k)].astype(numpy.int64, copy=False)


def filter_mask_from_attrs(attrs: Mapping[str, numpy.ndarray], fvalue) -> numpy.ndarray:
    """Boolean mask for ``[attr, op, value]`` or a list of triples from ``parse_predicate``."""
    from ann_benchmarks.algorithms.faiss.postfilter import filter_mask_from_fvalue

    if fvalue == ["No_filter"] or fvalue == "No_filter":
        n = len(next(iter(attrs.values())))
        return numpy.ones(n, dtype=bool)
    if _is_clause_list(fvalue):
        return predicate_mask(attrs, fvalue)
    arr = resolve_attr_array(attrs, fvalue[0])
    if not numpy.issubdtype(arr.dtype, numpy.number):
        return predicate_mask(attrs, [tuple(fvalue)])
    return filter_mask_from_fvalue(arr.astype(numpy.float32), fvalue)
