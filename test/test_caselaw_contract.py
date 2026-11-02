"""Unit tests for the CaseLaw integration contract (Section 17).

Runnable without the CaseLaw download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ann_benchmarks.algorithms.faiss.postfilter import filter_mask_from_fvalue
from ann_benchmarks.attrs import (
    filter_mask_from_attrs,
    parse_predicate,
    predicate_mask,
)
from ann_benchmarks.corpora import (
    CASELAW_N_BASE,
    build_sample_manifest,
    caselaw_recall_at_k,
    cardinality,
    draw_stratified_sample,
    dump_manifest_bytes,
)
from ann_benchmarks.definitions import Definition
from ann_benchmarks.results import build_result_filepath
from ann_benchmarks.runner import parse_filter


def _dummy_def(algo: str = "hnsw(faiss)") -> Definition:
    return Definition(
        algorithm=algo,
        docker_tag="ann-benchmarks-faiss",
        module="ann_benchmarks.algorithms.faiss_hnsw",
        constructor="FaissHNSW",
        arguments=["euclidean", {"M": 16, "efConstruction": 128}],
        query_argument_groups=[[40]],
        disabled=False,
    )


def test_parse_filter_unchanged_for_morevec():
    assert parse_filter("avg_rating >= 7.0") == ["avg_rating", ">=", "7.0"]
    assert parse_filter("year <= 2015") == ["year", "<=", "2015"]
    assert parse_filter("No_filter") == ["No_filter"]


def test_parse_predicate_clauses_and_quoted_spaces():
    p_date = parse_predicate("date_filed >= 693596 AND date_filed <= 700886")
    assert p_date == [
        ("date_filed", ">=", "693596"),
        ("date_filed", "<=", "700886"),
    ]

    p_court = parse_predicate('court_full_name = "Supreme Court"')
    assert p_court == [("court_full_name", "=", "Supreme Court")]

    p_conj = parse_predicate(
        'court_jurisdiction = "California" AND court_full_name = "Supreme Court"'
    )
    assert p_conj == [
        ("court_jurisdiction", "=", "California"),
        ("court_full_name", "=", "Supreme Court"),
    ]

    assert parse_predicate("No_filter") == ["No_filter"]


def test_predicate_mask_on_5_row_toy_table():
    attrs = {
        "date_filed": np.array([690000, 695000, 700000, 705000, 710000], dtype=np.int64),
        "court_jurisdiction": np.array(
            ["California", "New York", "California", "Texas", "California"], dtype=object
        ),
        "court_type": np.array(["A", "S", "A", "A", "F"], dtype=object),
        "court_full_name": np.array(
            [
                "Supreme Court",
                "Court of Appeals",
                "Supreme Court",
                "District Court",
                "Supreme Court",
            ],
            dtype=object,
        ),
        "avgrating": np.array([6.5, 7.0, 8.2, 5.1, 9.0], dtype=np.float64),
    }

    # String equality
    m_str = predicate_mask(attrs, parse_predicate('court_full_name = "Supreme Court"'))
    assert m_str.tolist() == [True, False, True, False, True]

    # Ordinal range
    m_ord = predicate_mask(
        attrs, parse_predicate("date_filed >= 695000 AND date_filed <= 705000")
    )
    assert m_ord.tolist() == [False, True, True, True, False]

    # Conjunction of string equality and ordinal range
    m_and = predicate_mask(
        attrs,
        parse_predicate(
            'court_jurisdiction = "California" AND date_filed >= 695000'
        ),
    )
    assert m_and.tolist() == [False, False, True, False, True]

    # filter_mask_from_attrs accepts list-of-triples as well as single triple
    assert np.array_equal(
        filter_mask_from_attrs(
            attrs,
            parse_predicate(
                'court_jurisdiction = "California" AND court_type = "A"'
            ),
        ),
        np.array([True, False, True, False, False], dtype=bool),
    )

    # MoRe single numeric triple still matches filter_mask_from_fvalue
    triple = parse_filter("avg_rating >= 7.0")
    m_more = predicate_mask(attrs, triple)
    m_fval = filter_mask_from_fvalue(attrs["avgrating"].astype(np.float32), triple)
    assert np.array_equal(m_more, m_fval)
    assert np.array_equal(filter_mask_from_attrs(attrs, triple), m_fval)


def test_sample_draw_determinism_quotas_and_bounds():
    rng = np.random.RandomState(0)
    sels = np.concatenate(
        [
            rng.uniform(0.005, 0.01, size=123),
            rng.uniform(0.01, 0.1, size=234),
            rng.uniform(0.1, 0.5, size=345),
            rng.uniform(0.0001, 0.004, size=111),
        ]
    )
    nres = np.full(len(sels), 10, dtype=np.int32)
    filters = ['court_type = "A"'] * len(sels)

    idx1, pop1, tr1 = draw_stratified_sample(sels, nres, frac=0.10, seed=42)
    idx2, pop2, tr2 = draw_stratified_sample(sels, nres, frac=0.10, seed=42)
    assert np.array_equal(idx1, idx2)
    assert pop1 == pop2
    assert tr1 == tr2
    assert sum(tr1.values()) == round(0.10 * len(sels))
    assert len(idx1) == round(0.10 * len(sels))

    m1 = build_sample_manifest(sels, nres, filters, frac=0.10, seed=42)
    m2 = build_sample_manifest(sels, nres, filters, frac=0.10, seed=42)
    assert dump_manifest_bytes(m1) == dump_manifest_bytes(m2)
    assert m1["n_train"] + m1["n_test"] == m1["n_population"] == len(sels)
    assert m1["n_sample"] == m1["n_train"]

    with pytest.raises(ValueError):
        draw_stratified_sample(sels, nres, frac=0.04, seed=42)
    with pytest.raises(ValueError):
        draw_stratified_sample(sels, nres, frac=0.21, seed=42)


def test_build_result_filepath_flex_and_native():
    defn = _dummy_def()
    flex_path = build_result_filepath(
        "glove-100-angular",
        10,
        defn,
        [40],
        batch_mode=False,
        filter_id=0,
        dataset_size="large",
        data_table="movies",
        att_idx=0,
        workload="flex",
    )
    assert flex_path.startswith("results/MoRe_UPD_")

    native_path = build_result_filepath(
        "glove-100-angular",
        10,
        defn,
        [40],
        batch_mode=False,
        filter_id=0,
        dataset_size="large",
        data_table="cases",
        att_idx=0,
        workload="native",
    )
    assert native_path.startswith("results/CaseLaw_7M_native_cases/")
    assert cardinality("caselaw", "cases") == CASELAW_N_BASE == 7_414_023


def test_caselaw_recall_short_gt():
    gt = [10, 20, 30, -1, -1, -1, -1, -1, -1, -1]
    assert caselaw_recall_at_k([10, 20, 99], gt, nres=3, k=10) == pytest.approx(2.0 / 3.0)
