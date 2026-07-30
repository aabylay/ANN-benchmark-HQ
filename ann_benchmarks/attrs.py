"""Attribute name → column resolution for multi-attr filtered ANN.

Shared by flex / hard / superhard workload modes. Maps filter strings such as
``avg_rating >= 7.0`` or ``year <= 2015`` onto the numeric columns loaded from
MoRe train HDF5s.
"""

from __future__ import annotations

from typing import Dict, Mapping, Union

import numpy

# Column order matches load_train_dataset stacking in runner.py.
MOVIES_ATTR_NAMES = ["avgrating", "title", "genre", "num_votes", "year"]
REVIEWS_ATTR_NAMES = ["mid", "uid", "likeshare", "movierating", "total_votes"]

# Numeric columns that may appear in filter predicates (SQL + FAISS masks).
MOVIES_NUMERIC_ATTRS = ["avgrating", "num_votes", "year"]
REVIEWS_NUMERIC_ATTRS = ["likeshare", "movierating", "total_votes"]

PRIMARY_ATTR = {
    "movies": "avgrating",
    "reviews": "total_votes",
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
}

ALLOWED_OPS = frozenset({">=", "<=", ">", "<", "=", "!="})

AttrsDict = Dict[str, numpy.ndarray]


def attr_names_for(dataset_type: str) -> list[str]:
    if dataset_type == "movies":
        return list(MOVIES_ATTR_NAMES)
    if dataset_type == "reviews":
        return list(REVIEWS_ATTR_NAMES)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def numeric_attr_names_for(dataset_type: str) -> list[str]:
    if dataset_type == "movies":
        return list(MOVIES_NUMERIC_ATTRS)
    if dataset_type == "reviews":
        return list(REVIEWS_NUMERIC_ATTRS)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def build_attrs_dict(X_attrs: numpy.ndarray, dataset_type: str) -> AttrsDict:
    """Build name→array map from stacked train attributes (shape 5 x N)."""
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
            out[name] = col.astype(numpy.float32)
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


def filter_mask_from_attrs(attrs: Mapping[str, numpy.ndarray], fvalue) -> numpy.ndarray:
    """Boolean mask for ``[attr, op, value]`` against the named column."""
    from ann_benchmarks.algorithms.faiss.postfilter import filter_mask_from_fvalue

    if fvalue == ["No_filter"] or fvalue == "No_filter":
        n = len(next(iter(attrs.values())))
        return numpy.ones(n, dtype=bool)
    arr = resolve_attr_array(attrs, fvalue[0])
    return filter_mask_from_fvalue(arr.astype(numpy.float32), fvalue)


def validate_op(op: str) -> str:
    if op not in ALLOWED_OPS:
        raise ValueError(f"Unsupported filter operator: {op}")
    return op
