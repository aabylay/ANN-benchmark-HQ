"""Shared multi-attribute table helpers for pgvector plans."""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy

from ann_benchmarks.attrs import (
    coerce_attrs,
    numeric_attr_names_for,
    sql_column_for_attr,
    validate_op,
)


def create_items_table(cur, dim: int, dataset_type: str, attrs: Mapping[str, numpy.ndarray]):
    """CREATE TABLE items with embedding + all numeric filter columns."""
    cols = numeric_attr_names_for(dataset_type)
    col_sql = ", ".join(f"{c} FLOAT" for c in cols)
    cur.execute("DROP TABLE IF EXISTS items")
    cur.execute(
        f"CREATE TABLE items (id int, embedding vector({dim}), {col_sql})"
    )
    cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
    return cols


def copy_items_rows(
    cur,
    X: numpy.ndarray,
    attrs: Mapping[str, numpy.ndarray],
    cols: Sequence[str],
):
    """COPY id, embedding, and numeric attribute columns."""
    type_list = ["int4", "vector"] + ["float8"] * len(cols)
    col_names = ", ".join(["id", "embedding"] + list(cols))
    with cur.copy(f"COPY items ({col_names}) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(type_list)
        for i, embedding in enumerate(X):
            row = [i, embedding.tolist()] + [float(attrs[c][i]) for c in cols]
            copy.write_row(tuple(row))


def prepare_attrs(X_attr, dataset_type: str):
    return coerce_attrs(X_attr, dataset_type)


def where_clause(filter_parts, dataset_type: str) -> str:
    """Build a safe WHERE clause from parsed ``[attr, op, value]``."""
    if filter_parts == ["No_filter"] or filter_parts == "No_filter":
        return ""
    attr, op, value = filter_parts[0], filter_parts[1], filter_parts[2]
    col = sql_column_for_attr(attr, dataset_type)
    op = validate_op(op)
    # value is a numeric literal from parse_filter; cast explicitly.
    float(value)  # validate
    return f"WHERE {col} {op} {value}::FLOAT"


def filtered_order_query(metric: str, filter_parts, dataset_type: str) -> str:
    if metric == "angular":
        dist = "<=>"
    elif metric == "euclidean":
        dist = "<->"
    else:
        raise RuntimeError(f"unknown metric {metric}")
    if filter_parts == ["No_filter"] or filter_parts == "No_filter":
        return f"""
            SELECT id
            FROM items
            ORDER BY embedding {dist} %s
            LIMIT %s
        """
    where = where_clause(filter_parts, dataset_type)
    return f"""
        SELECT id
        FROM items
        {where}
        ORDER BY embedding {dist} %s
        LIMIT %s
    """


def create_attr_indexes(cur, dataset_type: str):
    for col in numeric_attr_names_for(dataset_type):
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS items_{col}_idx ON items ({col})"
        )
        print(f"[PGVector] Created index on {col}")
