from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qo_prototype as qo  # noqa: E402
import query_optimizer_analysis as qoa  # noqa: E402


@pytest.fixture
def clean_output_dir():
    """Temporary synthetic outputs that are deleted as soon as each test ends."""
    with tempfile.TemporaryDirectory(prefix="fannsqo-qo-test-") as directory:
        yield Path(directory)


def _loaded_sweep(rho_shift: float = 0.0, n_queries: int = 2) -> pd.DataFrame:
    rows = []
    plan_hps = {
        "BF": [0],
        "HNSW-pre": [10, 20],
        "HNSW-post": [10, 20],
        "IVF-pre": [5, 10],
        "IVF-post": [5, 10],
    }
    for query_id in range(n_queries):
        sigma = 0.01 * (query_id + 1)
        for plan, hps in plan_hps.items():
            for hp in hps:
                rows.append({
                    "query_type": "movies",
                    "query_id_num": query_id,
                    "filter_id": 0,
                    "k": 10,
                    "plan": plan,
                    "hyperparam": hp,
                    "sigma": sigma,
                    "rho": 0.2 + rho_shift,
                    "N": float(qo.N_TABLE["movies"]),
                    "n_pass": sigma * qo.N_TABLE["movies"],
                    "recall": 1.0 if plan == "BF" else 0.8 + 0.01 * hp,
                    "runtime_ms": 10.0 if plan == "BF" else 1.0 + hp / 10.0,
                })
    return pd.DataFrame(rows)


def test_quantile_leaf_matches_empirical_quantile() -> None:
    y = np.arange(100, dtype=float)
    tree = qo.CART(task="quantile", quantile=0.9, max_depth=0, min_leaf=1)
    tree.fit(np.zeros((len(y), 1)), y)
    got = tree.predict(np.zeros((1, 1)))
    assert got[0] == np.quantile(y, 0.9, method="higher")

    low = qo.CART(task="quantile", quantile=0.2, max_depth=0, min_leaf=1).fit(
        np.zeros((len(y), 1)), y
    )
    high = qo.CART(task="quantile", quantile=0.8, max_depth=0, min_leaf=1).fit(
        np.zeros((len(y), 1)), y
    )
    assert high.predict(np.zeros((1, 1)))[0] >= low.predict(np.zeros((1, 1)))[0]


def test_pinball_split_separates_two_budgets() -> None:
    # Feature 0 perfectly separates a low-budget group from a high-budget group.
    x_low = np.column_stack([np.zeros(40), np.linspace(0, 1, 40)])
    x_high = np.column_stack([np.ones(40), np.linspace(0, 1, 40)])
    y = np.concatenate([np.full(40, 1.0), np.full(40, 5.0)])
    tree = qo.CART(task="quantile", quantile=0.8, max_depth=1, min_leaf=10)
    tree.fit(np.vstack([x_low, x_high]), y)
    low = tree.predict(np.array([[0.0, 0.2]]))[0]
    high = tree.predict(np.array([[1.0, 0.2]]))[0]
    assert low == 1.0
    assert high == 5.0


def test_tiered_features_use_three_correlations() -> None:
    rows = pd.DataFrame({
        "sigma": [0.1, 0.2],
        "rho": [0.0, 0.1],
        "rho_1": [-0.4, 0.2],
        "rho_10": [0.0, 0.1],
        "rho_100": [0.05, -0.05],
        "k": [10, 10],
    })
    feats = qo.stage12_feats(rows, qo.FEATURE_SIGMA_RHO_TIERS)
    assert feats.shape == (2, 5)
    block = qo.correlation_block(rows, qo.FEATURE_SIGMA_RHO_TIERS)
    stage3 = qo.stage3_feats(
        np.array([10.0, 20.0]),
        block,
        np.array([1000.0, 1000.0]),
        feature_set=qo.FEATURE_SIGMA_RHO_TIERS,
    )
    assert stage3.shape == (2, 5)
    assert qo.estimated_feature_set(
        ["gls_correlation_1", "gls_correlation_10", "gls_correlation_100"]
    ) == qo.FEATURE_SIGMA_RHO_TIERS
    assert qo.estimated_feature_set(["gls_correlation"]) == qo.FEATURE_SIGMA_RHO


def test_sigma_only_features_are_invariant_to_rho() -> None:
    rows = pd.DataFrame({
        "sigma": [0.01, 0.1],
        "rho": [-0.8, 0.8],
        "k": [10, 20],
    })
    changed = rows.assign(rho=[0.3, -0.4])

    np.testing.assert_array_equal(
        qo.stage12_feats(rows, qo.FEATURE_SIGMA_ONLY),
        qo.stage12_feats(changed, qo.FEATURE_SIGMA_ONLY),
    )
    assert not np.array_equal(
        qo.stage12_feats(rows, qo.FEATURE_SIGMA_RHO),
        qo.stage12_feats(changed, qo.FEATURE_SIGMA_RHO),
    )

    args = (
        np.array([100.0, 200.0]),
        rows["rho"].to_numpy(),
        np.array([1000.0, 1000.0]),
        np.array([10.0, 20.0]),
        rows["k"].to_numpy(),
    )
    changed_args = (args[0], changed["rho"].to_numpy(), *args[2:])
    np.testing.assert_array_equal(
        qo.stage3_feats(*args, qo.FEATURE_SIGMA_ONLY),
        qo.stage3_feats(*changed_args, qo.FEATURE_SIGMA_ONLY),
    )
    assert not np.array_equal(
        qo.stage3_feats(*args, qo.FEATURE_SIGMA_RHO),
        qo.stage3_feats(*changed_args, qo.FEATURE_SIGMA_RHO),
    )


def test_group_folds_do_not_depend_on_rho() -> None:
    sweep = _loaded_sweep()
    pp = sweep[qo.PAIR_KEYS + ["plan", "sigma", "rho"]].copy()
    changed = pp.sample(frac=1.0, random_state=7).assign(rho=lambda x: x["rho"] * -1)
    a = qo.group_folds(pp).sort_values(qo.PAIR_KEYS + ["plan"])["fold"].to_numpy()
    b = qo.group_folds(changed).sort_values(qo.PAIR_KEYS + ["plan"])["fold"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_sigma_only_cv_is_invariant_to_rho() -> None:
    def evaluate(sweep: pd.DataFrame):
        pairplan = qo.build_pairplan(sweep, recall_target=0.95)
        measurements = qo.build_measure(sweep)
        pairs = list(pairplan[qo.PAIR_KEYS].drop_duplicates().itertuples(index=False, name=None))
        oracle = qo.build_oracle(measurements, pairs, recall_target=0.95)
        records, _ = qo.run_cv(
            sweep,
            pairplan,
            measurements,
            oracle,
            qo.derive_grids(sweep),
            min_leaf=1,
            recall_target=0.95,
            feature_set=qo.FEATURE_SIGMA_ONLY,
        )
        return qo.optimizer_metrics(records, gamma=1.0, lam=0.0, recall_target=0.95)

    original = evaluate(_loaded_sweep(n_queries=10))
    changed = evaluate(_loaded_sweep(rho_shift=0.6, n_queries=10))
    assert original["chosen"] == changed["chosen"]
    np.testing.assert_array_equal(original["chosen_hyperparam"], changed["chosen_hyperparam"])
    assert original["qps"] == pytest.approx(changed["qps"])


def test_study_validation_requires_complete_identical_measurements() -> None:
    exact = _loaded_sweep()
    estimated = _loaded_sweep(rho_shift=0.1)
    qo.validate_study_pair(exact, estimated)

    incomplete = estimated.drop(estimated.index[-1])
    with pytest.raises(ValueError, match="missing|incomplete|row-key mismatch"):
        qo.validate_study_pair(exact, incomplete)

    mismatched = estimated.copy()
    mismatched.loc[0, "runtime_ms"] += 1.0
    with pytest.raises(ValueError, match="measurement mismatch"):
        qo.validate_study_pair(exact, mismatched)

    inconsistent = exact.copy()
    inconsistent.loc[0, "sigma"] *= 2
    with pytest.raises(ValueError, match="pair-level metadata"):
        qo.validate_study_sweep(inconsistent)


def test_pooled_tables_may_have_different_action_grids() -> None:
    movies = _loaded_sweep()
    reviews = _loaded_sweep().assign(
        query_type="reviews",
        N=float(qo.N_TABLE["reviews"]),
    )
    reviews["n_pass"] = reviews["sigma"] * reviews["N"]
    extra = reviews[
        reviews["plan"].isin(["IVF-pre", "IVF-post"]) & (reviews["hyperparam"] == 10)
    ].copy()
    extra["hyperparam"] = 20
    extra["recall"] = 1.0
    pooled = pd.concat([movies, reviews, extra], ignore_index=True)

    qo.validate_study_sweep(pooled)
    global_grids = qo.derive_grids(pooled)
    dataset_grids = qo.derive_dataset_grids(pooled)
    snapped = qo.snap_by_dataset(
        np.array([20.0, 20.0]),
        np.array(["movies", "reviews"]),
        np.array([10, 10]),
        "IVF-pre",
        global_grids,
        dataset_grids,
        force_positive=True,
    )
    np.testing.assert_array_equal(snapped, [10.0, 20.0])


def test_measurement_resolution_and_safety_bump() -> None:
    key = ("movies", 0, 0, 10, "HNSW-pre")
    meas = {
        key: (
            np.array([10.0, 20.0, 40.0]),
            np.array([0.8, 0.9, 1.0]),
            np.array([1.0, 2.0, 4.0]),
        )
    }
    assert qo.resolve_measurement(meas, key, 20) == (20.0, 0.9, 2.0)
    assert qo.resolve_measurement(meas, key, 11) == (20.0, 0.9, 2.0)
    assert qo.resolve_measurement(meas, key, 80) == (40.0, 1.0, 4.0)
    assert qo.bump_one(20, [10, 20, 40]) == 40
    assert qo.bump_one(40, [10, 20, 40]) == 40


def test_repeated_action_measurements_are_averaged() -> None:
    sweep = _loaded_sweep()
    repeated = pd.concat([sweep, sweep.iloc[[0]].assign(runtime_ms=30.0)], ignore_index=True)
    collapsed = qo.collapse_repeated_measurements(repeated)
    key = sweep.iloc[0][qo.STUDY_ROW_KEYS]
    mask = np.logical_and.reduce([
        collapsed[column].eq(value).to_numpy()
        for column, value in key.items()
    ])
    row = collapsed.loc[mask].iloc[0]
    assert row["runtime_ms"] == pytest.approx((10.0 + 30.0) / 2.0)
    assert row["measurement_repeats"] == 2


def test_selection_frame_records_requested_and_effective_hyperparams() -> None:
    pair = ("movies", 0, 0, 10)
    record = {
        "pair": pair,
        "cands": {
            "BF": {
                "p_feas": 1.0,
                "cost": 10.0,
                "cost_b": 10.0,
                "hp": 0.0,
                "hp_b": 0.0,
                "effective_hp": 0.0,
                "effective_hp_b": 0.0,
                "rt": 10.0,
                "rt_b": 10.0,
                "rec": 1.0,
                "rec_b": 1.0,
            },
            "HNSW-pre": {
                "p_feas": 0.9,
                "cost": 2.0,
                "cost_b": 4.0,
                "hp": 20.0,
                "hp_b": 40.0,
                "effective_hp": 20.0,
                "effective_hp_b": 40.0,
                "rt": 2.0,
                "rt_b": 4.0,
                "rec": 0.9,
                "rec_b": 1.0,
            },
        },
        "oracle_plan": "HNSW-pre",
        "oracle_cost": 2.0,
        "oracle_recall": 0.9,
        "bf_rt": 10.0,
        "hnsw": {"rt": 2.0, "rec": 0.9},
        "ivf": {"rt": 5.0, "rec": 0.95},
    }
    metrics = qo.optimizer_metrics([record], 1.0, 0.0, False, recall_target=0.9)
    bumped = qo.optimizer_metrics([record], 1.0, 0.0, True, recall_target=0.9)
    selection = qo.build_selection_frame(
        [record],
        metrics,
        {pair: (0.01, 0.2, "movies")},
        bumped,
        "sigma_only",
        qo.FEATURE_SIGMA_ONLY,
    )

    assert selection.loc[0, "chosen"] == "HNSW-pre"
    assert selection.loc[0, "chosen_hyperparam"] == 20.0
    assert selection.loc[0, "effective_hyperparam"] == 20.0
    assert selection.loc[0, "safety_bump_hyperparam"] == 40.0
    assert selection.loc[0, "safety_bump_effective_hyperparam"] == 40.0


def test_three_way_matched_recall_outputs(clean_output_dir: Path) -> None:
    tmp_path = clean_output_dir
    recalls = [0.90, 0.95, 1.0]

    def frame(qps):
        return pd.DataFrame({
            "gamma": [1.0, 2.0, 3.0],
            "avg_recall": recalls,
            "qps": qps,
        })

    results = {
        "sigma_only": {"pareto": frame([100.0, 80.0, 60.0]), "feature_set": qo.FEATURE_SIGMA_ONLY},
        "estimated": {"pareto": frame([120.0, 96.0, 72.0]), "feature_set": qo.FEATURE_SIGMA_RHO},
        "exact": {"pareto": frame([130.0, 104.0, 78.0]), "feature_set": qo.FEATURE_SIGMA_RHO},
    }
    _, matched = qo.write_gls_contribution_outputs(results, tmp_path, recall_target=0.95)

    at_target = matched[np.isclose(matched["recall_level"], 0.95)].iloc[0]
    assert at_target["estimated_vs_sigma_speedup"] == pytest.approx(1.2)
    assert at_target["estimated_vs_exact_qps"] == pytest.approx(96.0 / 104.0)
    interpolated = matched[np.isclose(matched["recall_level"], 0.925)].iloc[0]
    assert interpolated["sigma_only_qps"] == pytest.approx(1.0 / (0.5 / 100.0 + 0.5 / 80.0))
    assert interpolated["sigma_only_mix_high"] == pytest.approx(0.5)
    assert (tmp_path / "gls_contribution_pareto.csv").is_file()
    assert (tmp_path / "gls_contribution_matched_recall.csv").is_file()
    assert (tmp_path / "gls_contribution_pareto.png").is_file()

    summary = pd.DataFrame({
        "workload": ["hard_movies", "hard_movies"],
        "recall_target": [0.90, 0.95],
        "estimated_vs_sigma_speedup": [1.1, 1.2],
        "exact_vs_sigma_speedup": [1.15, 1.25],
    })
    qo.plot_gls_contribution_summary(summary, tmp_path / "gls_contribution_summary.png")
    assert (tmp_path / "gls_contribution_summary.png").is_file()


def test_safety_pareto_removes_dominated_points(clean_output_dir: Path) -> None:
    points = pd.DataFrame({
        "gamma": [1.0, 2.0, 3.0, 4.0, 5.0],
        "avg_recall": [0.90, 0.94, 0.93, 0.89, 0.97],
        "recall_fail_rate": [0.20, 0.10, 0.10, 0.20, 0.05],
        "qps": [100.0, 90.0, 80.0, 95.0, 50.0],
    })

    frontier = qo.safety_pareto_envelope_frame(points)

    assert frontier["gamma"].tolist() == [5.0, 2.0, 1.0]
    assert frontier["recall_fail_rate"].tolist() == pytest.approx([0.05, 0.10, 0.20])
    output = clean_output_dir / "safety.png"
    qo.plot_safety_pareto(points, output, "test", recall_target=0.95)
    assert output.is_file()


def test_hyperparam_tuning_uses_aggregate_workload_qps() -> None:
    rows = []
    for hyperparam, runtimes in ((10, [0.001, 1.0]), (20, [0.01, 0.01])):
        for runtime in runtimes:
            rows.append({
                "algorithm": "hnsw(faiss)",
                "hyperparam": hyperparam,
                "hyperparam_name": "ef_search",
                "recall": 0.95,
                "runtime": runtime,
                "qps": 1.0 / runtime,
            })
    result = qoa.find_best_hyperparams(pd.DataFrame(rows), recall_target=0.95)

    # mean(1 / latency) would incorrectly pick hp=10; n / sum(latency) picks hp=20.
    assert result.iloc[0]["hyperparam"] == 20
    assert result.iloc[0]["mean_qps"] == pytest.approx(100.0)


def test_run_one_writes_auditable_selection(clean_output_dir: Path) -> None:
    tmp_path = clean_output_dir
    result = qo.run_one(
        Path("in-memory-test-sweep"),
        "sigma_only",
        tmp_path,
        gamma=1.3,
        lam=4.0,
        min_leaf=1,
        recall_target=0.95,
        feature_set=qo.FEATURE_SIGMA_ONLY,
        sweep=_loaded_sweep(n_queries=10),
    )

    selection = pd.read_csv(tmp_path / "selection_sigma_only.csv")
    assert len(selection) == 10
    assert {
        "chosen_hyperparam",
        "effective_hyperparam",
        "safety_bump_hyperparam",
        "safety_bump_effective_hyperparam",
    }.issubset(selection.columns)
    assert result["feature_set"] == qo.FEATURE_SIGMA_ONLY
    plan_table = pd.read_csv(tmp_path / "qps_recall_plans_sigma_only.csv")
    assert set(plan_table.loc[plan_table["kind"] == "point", "name"]) == {
        "always-BF",
        "oracle",
        "optimizer",
        "physical-HNSW-pre",
        "physical-IVF-pre",
    }
    assert set(plan_table.loc[plan_table["kind"] == "curve", "plan"]) == {"HNSW-pre", "IVF-pre"}
    assert (tmp_path / "qps_recall_plans_sigma_only.png").is_file()


def test_static_curve_and_physical_optimizer_use_aggregate_qps() -> None:
    pair_a = ("movies", 0, 0, 10)
    pair_b = ("movies", 1, 0, 10)
    meas = {
        (*pair_a, "HNSW-pre"): (
            np.array([10.0, 20.0]),
            np.array([0.5, 1.0]),
            np.array([1.0, 3.0]),
        ),
        (*pair_b, "HNSW-pre"): (
            np.array([10.0, 20.0, 40.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 3.0, 8.0]),
        ),
    }
    curve = qo.static_plan_curve(meas, [pair_a, pair_b], "HNSW-pre")
    # hp=40 is not swept for pair_a, so it is not a common policy.
    assert curve["hyperparam"].tolist() == [10, 20]
    at_10 = curve[curve["hyperparam"] == 10].iloc[0]
    at_20 = curve[curve["hyperparam"] == 20].iloc[0]
    assert at_10["avg_recall"] == pytest.approx(0.75)
    assert at_10["qps"] == pytest.approx(1000.0)  # 2 queries / 0.002 s
    assert at_20["avg_recall"] == pytest.approx(1.0)
    assert at_20["qps"] == pytest.approx(2.0 / 0.006)

    records = [
        {"pair": pair_a, "phys": {"HNSW-pre": {"rt": 1.0, "rec": 0.5, "hp": 10.0}}},
        {"pair": pair_b, "phys": {"HNSW-pre": {"rt": 1.0, "rec": 1.0, "hp": 10.0}}},
    ]
    metrics = qo.physical_optimizer_metrics(records, "HNSW-pre", recall_target=0.95)
    assert metrics["qps"] == pytest.approx(1000.0)
    assert metrics["avg_recall"] == pytest.approx(0.75)
    assert metrics["recall_fail_rate"] == pytest.approx(0.5)
    assert metrics["hp_name"] == "efSearch"
