#!/usr/bin/env python3
"""FANNS query optimizer -- v2 prototype (FAISS only).

Design: prompts/qo_prototype.txt.

Decide, per (query vector, filter), which FAISS plan to run AND with which search
hyperparameter, using only cheap plan-time inputs (sigma, optional rho, N) -- no
vector search. Unlike v1 (which ramped to the tuned hp online and charged the
ramp), v2 PREDICTS the hyperparameter directly, so the cost we compare is the
single tuned query. That introduces a real recall-miss risk (an under-predicted
hp can miss recall at query time), which the decision rule and the evaluation
handle explicitly.

Pipeline (three per-plan models + one decision rule):

  STAGE 1  feasibility (per ANN plan, classifier). CART (depth <= 3) on
      X = {sigma, rho, k} (pooled across tables) -> P_feasible in [0,1]. k enters
      because feasibility is per-query-k: a larger k needs a larger hp and shrinks
      the feasible region within the swept grid.

  STAGE 2  hyperparameter (per ANN plan). "Best hp" = the LOWEST swept hp that
      reaches recall >= the recall target (default 0.95; campaign also 0.80 /
      0.90 / 0.99), evaluated PER k. Two variants are fit on X = {sigma, rho, k}:
        * ablation (used by the optimizer): a log(hp) regressor on FEASIBLE rows
          only, snapped/clamped to the grid -- "steadier at the boundary". An
          optional one-step SAFETY BUMP raises the predicted hp to cut misses.
        * unified (reported for reference): a single 0-sentinel regressor whose
          feasibility is derived as (predicted hp > 0), so Stages 1+2 collapse
          to one model. It over-predicts feasibility at the boundary, which is
          why the optimizer gates on the Stage-1 classifier instead.

  STAGE 3  runtime (per plan, ms). log-linear OLS predicting log(runtime_ms)
      from log(n_pass), rho, log(N), log(hp) and log(k) (BF has no hp term).
      log(k) matters for the POST-filter plans, whose over-fetch pool
      search_k = min(1000, ceil(k/sigma)) and effective efSearch = max(hp,
      search_k) both scale with k; for the pre-filter plans and BF, runtime is
      ~k-independent and log(k) simply gets a near-zero coefficient. Trained on
      the FULL sweep so it learns the hp->runtime surface; fed the Stage-2
      predicted hp* at inference (teacher-forced).

  STAGE 4  robustness-aware decision rule.
      Candidate set = ANN plans with P_feasible >= 0.5 + BF (always included).
      score_p = cost_p * m_p, with m_BF = 1 and
          m_ANN = gamma * (1 + lambda * (1 - P_feasible_p)).
      Pick argmin score; gamma >= 1 is the index margin protecting exact BF
      (recall = 1.0), lambda penalises low-confidence ANN picks. Sweep gamma to
      generate operating points, then retain the non-dominated
      latency-vs-recall-safety Pareto envelope.

Evaluation is grouped 5-fold CV (folds split *queries*, so a query's filters
never straddle train/test). The realised RECALL-FAILURE RATE (fraction of chosen
pairs whose measured recall < the recall target, default 0.95) is the new, critical v2
metric. Runs on both exact and estimated GLS. Outputs -> plots/qo_prototype/{rXXX}/.

``qps_recall_plans_{variant}.png`` is the throughput-vs-recall figure: static
HNSW-pre and IVF-pre curves (one global efSearch or nprobe), plus single points
for brute force (recall 1.0), the per-query oracle, the full query optimizer,
and physical-only optimizers that keep the logical plan fixed (HNSW-pre or
IVF-pre) and only predict the search hyperparameter.

The opt-in ``--gls-contribution-study`` mode runs three variants on identical
held-out measurements and folds: selectivity-only, estimated GLS, and exact GLS.
It is an offline replay evaluation: the selected discrete (plan, hyperparameter)
is looked up in benchmark measurements rather than executed again. Reported QPS
is aggregate search-only throughput; a paper-level end-to-end result must also
charge GLS estimation and optimizer inference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_optimizer_analysis import (  # noqa: E402
    K10_RECALL_FOOTNOTE,
    RECALL_TARGET,
    apply_k10_footnote,
    df_is_k10_only,
    format_recall_target,
    recall_tag,
)
N_TABLE = {"movies": 551_155, "reviews": 2_598_267}


def cardinality_for(query_type: str) -> float:
    """Movies and reviews stay on the literal ``N_TABLE`` fallbacks.

    ``cases`` is looked up in the corpus registry (N = 7_414_023).
    """
    if query_type in N_TABLE:
        return float(N_TABLE[query_type])
    if query_type == "cases":
        from ann_benchmarks.corpora import cardinality
        return float(cardinality("caselaw", "cases"))
    raise ValueError(f"unsupported query_type {query_type!r}")


def supported_query_types() -> set[str]:
    return set(N_TABLE) | {"cases"}

PLANS = {
    "faiss-flat": "BF",
    "hnsw(faiss)": "HNSW-pre",
    "hnsw(faiss)-post": "HNSW-post",
    "faiss-ivf": "IVF-pre",
    "faiss-ivf-post": "IVF-post",
}
ANN_PLANS = ["HNSW-pre", "HNSW-post", "IVF-pre", "IVF-post"]
ALL_PLANS = ["BF"] + ANN_PLANS

PLAN_PLOT_COLORS = {
    "BF": "#e2585f",
    "HNSW-pre": "#5aa0ff",
    "HNSW-post": "#8e6bb0",
    "IVF-pre": "#39c07a",
    "IVF-post": "#e6a23c",
}

# Logical plans whose only learned choice is the search hyperparameter.
# HNSW-pre sweeps efSearch; IVF-pre (IVFFlat) sweeps nprobe.
PHYSICAL_PLANS = ("HNSW-pre", "IVF-pre")
PHYSICAL_PLAN_HP = {"HNSW-pre": "efSearch", "IVF-pre": "nprobe"}
PHYSICAL_CURVE_STYLE = {
    "HNSW-pre": {"color": "#e6a23c", "marker": "^"},
    "IVF-pre": {"color": "#2a9d8f", "marker": "v"},
}

# Flex-era fallback grids. Per-plan grids are normally DERIVED from the loaded
# sweep CSV (see derive_grids); these constants are used only when a plan has
# no swept rows. Hardcoding them as the working grids broke hard/superhard
# packs (swept efSearch grid started at 40, so predictions snapped to >= 100
# and measure_at clamped every HNSW measurement to the largest swept ef).
HNSW_GRID = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
             1200, 1400, 1600, 1800, 2000, 2250, 2500]
# IVF probes: the swept large-dataset grid, plus 500 = the reviews-only x2
# doubling of the 250 max applied in runner.py.
IVF_GRID = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            120, 140, 160, 180, 200, 225, 250, 500]
FALLBACK_GRID = {"HNSW-pre": HNSW_GRID, "HNSW-post": HNSW_GRID,
                 "IVF-pre": IVF_GRID, "IVF-post": IVF_GRID}


def derive_grids(sweep: pd.DataFrame) -> dict:
    """Per-plan search-hp grids taken from the swept hyperparams in the CSV."""
    grids = {}
    for plan in ANN_PLANS:
        vals = sorted(
            int(v)
            for v in sweep.loc[sweep["plan"] == plan, "hyperparam"].unique()
            if v > 0
        )
        grids[plan] = vals if vals else list(FALLBACK_GRID[plan])
    return grids


def derive_dataset_grids(sweep: pd.DataFrame) -> dict:
    """Per-(dataset, k, plan) action grids for pooled and multi-k workloads."""
    grids = {}
    global_grids = derive_grids(sweep)
    requests = sweep[["query_type", "k"]].drop_duplicates()
    for query_type, k in requests.itertuples(index=False, name=None):
        for plan in ANN_PLANS:
            vals = sorted(
                int(v)
                for v in sweep.loc[
                    (sweep["query_type"] == query_type)
                    & (sweep["k"] == k)
                    & (sweep["plan"] == plan),
                    "hyperparam",
                ].unique()
                if v > 0
            )
            grids[(query_type, int(k), plan)] = vals if vals else list(global_grids[plan])
    return grids


# A (query, filter, k) triple is the unit of decision: k is part of the query
# request (top-k), so it identifies a pair alongside the query vector and filter.
PAIR_KEYS = ["query_type", "query_id_num", "filter_id", "k"]

# wide enough to expose the Pareto: in v2 ANN is far cheaper than BF, so small
# gamma never displaces BF -- only large gamma trades latency for recall safety.
# Dense log-spaced grid: gamma is scored post-hoc from cached records (no model
# refit), so extra points are cheap and smooth out the frontier.
GAMMA_SWEEP = [round(float(g), 2) for g in np.geomspace(1.0, 100.0, 31)]

FEATURE_SIGMA_RHO = "sigma_rho"
FEATURE_SIGMA_ONLY = "sigma_only"
FEATURE_SIGMA_RHO_TIERS = "sigma_rho_tiers"
FEATURE_SETS = {FEATURE_SIGMA_RHO, FEATURE_SIGMA_ONLY, FEATURE_SIGMA_RHO_TIERS}
# Nested plan-time correlations: nearest 1, 10, and 100 partitions.
RHO_TIER_COLUMNS = ("rho_1", "rho_10", "rho_100")
GLS_TIER_SOURCE_COLUMNS = (
    "gls_correlation_1", "gls_correlation_10", "gls_correlation_100",
)

VARIANT_DISPLAY = {
    "sigma_only": "Selectivity only (sigma)",
    "estimated": "Estimated GLS (sigma, rho tiers 1/10/100)",
    "exact": "Exact GLS reference (ground-truth rho)",
}
VARIANT_COLORS = {
    "sigma_only": "#666666",
    "estimated": "#5aa0ff",
    "exact": "#39c07a",
}

SWEEP_REQUIRED_COLUMNS = {
    "algorithm",
    "query_type",
    "query_id_num",
    "filter_id",
    "k",
    "filter_selectivity",
    "gls_correlation",
    "hyperparam",
    "recall",
    "runtime",
}
STUDY_ROW_KEYS = PAIR_KEYS + ["plan", "hyperparam"]


# Nominal miss budgets for the pinball search-budget model. The tree predicts
# the (1 - delta) quantile of the minimum sufficient hyperparameter, so a
# smaller delta is a more conservative budget. 0.10 is the reported default.
DELTA_SWEEP = [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02]
DEFAULT_DELTA = 0.10


def _pinball_sum(y, yhat, q):
    """Sum of pinball (quantile) losses of predicting ``yhat`` at level ``q``."""
    residual = np.asarray(y, float) - float(yhat)
    # Under-prediction (y > yhat) is weighted by q; over-prediction by 1 - q.
    return float(np.where(residual >= 0.0, q * residual, (q - 1.0) * residual).sum())


def budget_sentinel(grids) -> float:
    """Log-budget above every real grid value. Encodes an infeasible plan."""
    max_hp = max(max(grid) for grid in grids.values())
    return float(np.log(max_hp) + 1.0)


# --------------------------------------------------------------------------- #
# tiny numpy CART (no sklearn dependency -- the glscore env has none)
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("feat", "thr", "left", "right", "val")

    def __init__(self):
        self.feat = -1
        self.thr = 0.0
        self.left = None
        self.right = None
        self.val = 0.0


class CART:
    """Minimal CART.

    task='reg' uses variance splits and a mean leaf. task='clf' uses Gini and a
    frequency leaf. task='quantile' minimises pinball loss at ``quantile`` and
    stores that empirical quantile in the leaf (numpy method='higher').
    """

    def __init__(self, task="reg", max_depth=3, min_leaf=30, quantile=0.5):
        self.task = task
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.quantile = float(quantile)
        self.root = None

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y, float), 0)
        return self

    def _leaf(self, y):
        node = _Node()
        if len(y) == 0:
            node.val = 0.0
        elif self.task == "quantile":
            node.val = float(np.quantile(y, self.quantile, method="higher"))
        else:
            node.val = float(y.mean())
        return node

    def _best_quantile_split(self, X, y):
        """Pinball-loss split. Candidate thresholds are a stride of the feature."""
        n, d = X.shape
        q = self.quantile
        best = None
        n_cand = 64
        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs = X[order, f]
            ys = y[order]
            lo, hi = self.min_leaf, n - self.min_leaf
            if hi < lo:
                continue
            cands = np.unique(np.round(
                np.linspace(lo, hi, num=min(n_cand, hi - lo + 1))
            ).astype(int))
            for i in cands:
                if xs[i - 1] == xs[i]:
                    continue
                left, right = ys[:i], ys[i:]
                loss = (
                    _pinball_sum(left, np.quantile(left, q, method="higher"), q)
                    + _pinball_sum(right, np.quantile(right, q, method="higher"), q)
                )
                if best is None or loss < best[0]:
                    best = (loss, f, 0.5 * (float(xs[i - 1]) + float(xs[i])))
        return best

    def _best_split(self, X, y):
        if self.task == "quantile":
            return self._best_quantile_split(X, y)
        n, d = X.shape
        best = None
        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs = X[order, f]
            ys = y[order]
            csum = np.cumsum(ys)
            total = csum[-1]
            nl = np.arange(1, n)
            nr = n - nl
            sl = csum[:-1]
            sr = total - sl
            if self.task == "reg":
                csq = np.cumsum(ys * ys)
                sql = csq[:-1]
                sqr = csq[-1] - sql
                imp = (sql - sl * sl / nl) + (sqr - sr * sr / nr)
            else:
                pl = sl / nl
                pr = sr / nr
                imp = nl * (2 * pl * (1 - pl)) + nr * (2 * pr * (1 - pr))
            valid = (xs[:-1] != xs[1:]) & (nl >= self.min_leaf) & (nr >= self.min_leaf)
            if not valid.any():
                continue
            imp = np.where(valid, imp, np.inf)
            i = int(np.argmin(imp))
            if best is None or imp[i] < best[0]:
                best = (float(imp[i]), f, 0.5 * (xs[i] + xs[i + 1]))
        return best

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or y.min() == y.max():
            return self._leaf(y)
        split = self._best_split(X, y)
        if split is None:
            return self._leaf(y)
        _, feat, thr = split
        mask = X[:, feat] <= thr
        if mask.sum() < self.min_leaf or (~mask).sum() < self.min_leaf:
            return self._leaf(y)
        node = _Node()
        node.feat = feat
        node.thr = thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.empty(len(X))
        for i in range(len(X)):
            node = self.root
            while node.feat >= 0:
                node = node.left if X[i, node.feat] <= node.thr else node.right
            out[i] = node.val
        return out


class LogOLS:
    """OLS on standardised features predicting log(y); exp() on predict."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        self.w, *_ = np.linalg.lstsq(Z, np.log(np.clip(y, 1e-9, None)), rcond=None)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return np.exp(Z @ self.w)


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def _avg_ranks(sorted_vals):
    n = len(sorted_vals)
    ranks = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[i : j + 1] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return ranks


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = _avg_ranks(s[order])
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def log_r2(y_true_ms, y_pred_ms):
    yt = np.log(np.clip(np.asarray(y_true_ms, float), 1e-9, None))
    yp = np.log(np.clip(np.asarray(y_pred_ms, float), 1e-9, None))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
def _source_label(source) -> str:
    return str(source) if source is not None else "sweep"


def validate_study_sweep(sweep: pd.DataFrame, source=None) -> None:
    """Validate one loaded sweep before using it in the GLS contribution study."""
    label = _source_label(source)
    if sweep.empty:
        raise ValueError(f"{label}: no supported FAISS rows")

    dup = sweep.duplicated(STUDY_ROW_KEYS, keep=False)
    if dup.any():
        examples = sweep.loc[dup, STUDY_ROW_KEYS].head(5).to_dict("records")
        raise ValueError(f"{label}: duplicate plan measurements for keys {examples}")

    numeric = [
        "query_id_num",
        "filter_id",
        "sigma",
        "rho",
        "N",
        "hyperparam",
        "recall",
        "runtime_ms",
        "k",
    ]
    for col in numeric:
        values = pd.to_numeric(sweep[col], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: column {col!r} contains missing or non-finite values")
    if not sweep["sigma"].between(0.0, 1.0).all():
        raise ValueError(f"{label}: selectivity must be in [0, 1]")
    if not sweep["rho"].between(-1.0, 1.0).all():
        raise ValueError(f"{label}: GLS correlation must be in [-1, 1]")
    if not sweep["recall"].between(0.0, 1.0).all():
        raise ValueError(f"{label}: recall must be in [0, 1]")
    if not (sweep["runtime_ms"] > 0).all():
        raise ValueError(f"{label}: runtime must be positive")
    if not (sweep["k"] > 0).all():
        raise ValueError(f"{label}: k must be positive")
    if not (sweep["hyperparam"] >= 0).all():
        raise ValueError(f"{label}: hyperparameters must be non-negative")

    meta_cols = ["sigma", "rho", "N", "n_pass"]
    meta_cols += [col for col in RHO_TIER_COLUMNS if col in sweep.columns]
    if "split" in sweep.columns:
        meta_cols.append("split")
        splits = set(sweep["split"].dropna().astype(str).unique())
        if not splits <= {"train", "test"}:
            raise ValueError(f"{label}: invalid split values {sorted(splits)}")
        tr_q = set(sweep.loc[sweep["split"] == "train", "query_id_num"].astype(int))
        te_q = set(sweep.loc[sweep["split"] == "test", "query_id_num"].astype(int))
        overlap = tr_q & te_q
        if overlap:
            raise ValueError(f"{label}: train/test query_id_num overlap {sorted(overlap)[:5]}")
    ordered = sweep.sort_values(PAIR_KEYS, kind="mergesort")
    pair_change = np.zeros(len(ordered), dtype=bool)
    if len(ordered) > 1:
        for pk in PAIR_KEYS:
            arr = ordered[pk].to_numpy()
            pair_change[1:] |= arr[1:] != arr[:-1]
    for col in meta_cols:
        arr = ordered[col].to_numpy()
        val_change = np.zeros(len(ordered), dtype=bool)
        if len(ordered) > 1:
            if np.issubdtype(arr.dtype, np.floating):
                both_nan = np.isnan(arr[1:]) & np.isnan(arr[:-1])
                val_change[1:] = (arr[1:] != arr[:-1]) & ~both_nan
            else:
                val_change[1:] = arr[1:] != arr[:-1]
        bad_idx = np.flatnonzero(val_change & ~pair_change)
        if len(bad_idx):
            examples = [
                tuple(ordered.iloc[int(i)][PAIR_KEYS])
                for i in bad_idx[:5]
            ]
            raise ValueError(
                f"{label}: pair-level metadata {col!r} varies across plans/hyperparameters; "
                f"examples={examples}"
            )

    plans = set(sweep["plan"].unique())
    missing_plans = set(ALL_PLANS) - plans
    if missing_plans:
        raise ValueError(f"{label}: missing required plans {sorted(missing_plans)}")

    all_pairs = {
        tuple(row)
        for row in sweep[PAIR_KEYS].drop_duplicates().itertuples(index=False, name=None)
    }
    dataset_grids = derive_dataset_grids(sweep)
    for plan in ALL_PLANS:
        plan_rows = sweep[sweep["plan"] == plan]
        plan_pairs = {
            tuple(row)
            for row in plan_rows[PAIR_KEYS].drop_duplicates().itertuples(index=False, name=None)
        }
        missing_pairs = all_pairs - plan_pairs
        if missing_pairs:
            examples = sorted(missing_pairs, key=str)[:5]
            raise ValueError(
                f"{label}: plan {plan} is missing {len(missing_pairs)} query pairs; examples={examples}"
            )

        bad = []
        counts = plan_rows.groupby(PAIR_KEYS, sort=False)["hyperparam"].nunique()
        for (qt, kk), g_cnt in counts.groupby(level=[0, 3]):
            expected_set = (
                {0.0}
                if plan == "BF"
                else {float(v) for v in dataset_grids[(qt, int(kk), plan)]}
            )
            sub_rows = plan_rows[(plan_rows["query_type"] == qt) & (plan_rows["k"] == kk)]
            if (g_cnt == len(expected_set)).all() and set(sub_rows["hyperparam"].astype(float).unique()) == expected_set:
                continue
            expected = tuple(sorted(expected_set))
            for pair, rows in sub_rows.groupby(PAIR_KEYS, sort=False):
                observed = tuple(sorted(float(v) for v in rows["hyperparam"].unique()))
                if observed != expected:
                    bad.append((tuple(pair), expected, observed))
                    if len(bad) == 5:
                        break
            if len(bad) == 5:
                break
        if bad:
            raise ValueError(f"{label}: incomplete {plan} hyperparameter grid; examples={bad}")


def validate_study_pair(exact: pd.DataFrame, estimated: pd.DataFrame) -> None:
    """Require exact- and estimated-GLS files to describe identical measurements."""
    validate_study_sweep(exact, "exact GLS sweep")
    validate_study_sweep(estimated, "estimated GLS sweep")

    left = exact.sort_values(STUDY_ROW_KEYS, kind="mergesort").reset_index(drop=True)
    right = estimated.sort_values(STUDY_ROW_KEYS, kind="mergesort").reset_index(drop=True)
    keys_match = len(left) == len(right) and all(
        left[col].equals(right[col]) for col in STUDY_ROW_KEYS
    )
    if not keys_match:
        left_keys = list(left[STUDY_ROW_KEYS].itertuples(index=False, name=None))
        right_keys = list(right[STUDY_ROW_KEYS].itertuples(index=False, name=None))
        left_set, right_set = set(left_keys), set(right_keys)
        only_exact = sorted(left_set - right_set, key=str)[:5]
        only_estimated = sorted(right_set - left_set, key=str)[:5]
        raise ValueError(
            "exact/estimated GLS row-key mismatch: "
            f"only_exact={only_exact}, only_estimated={only_estimated}"
        )

    for col in ("sigma", "N", "recall", "runtime_ms"):
        a = left[col].to_numpy(float)
        b = right[col].to_numpy(float)
        equal = np.isclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=False)
        if not equal.all():
            i = int(np.flatnonzero(~equal)[0])
            key = left.loc[i, STUDY_ROW_KEYS].to_dict()
            raise ValueError(
                f"exact/estimated GLS measurement mismatch for {col!r} at {key}: "
                f"{a[i]} != {b[i]}"
            )


def collapse_repeated_measurements(sweep: pd.DataFrame) -> pd.DataFrame:
    """Average repeated executions of the same discrete optimizer action."""
    repeated = sweep.duplicated(STUDY_ROW_KEYS, keep=False)
    if not repeated.any():
        sweep = sweep.copy()
        sweep["measurement_repeats"] = 1
        return sweep

    metadata = [
        column
        for column in ("sigma", "rho", "N", "n_pass", "m", "ef_construction",
                       "clusters", "split", *RHO_TIER_COLUMNS)
        if column in sweep
    ]
    repeated_rows = sweep.loc[repeated]
    variation = repeated_rows.groupby(
        STUDY_ROW_KEYS, sort=False, dropna=False
    )[metadata].nunique(dropna=False)
    if variation.gt(1).any(axis=None):
        raise ValueError("repeated action measurements have inconsistent metadata")

    aggregations = {
        column: "first"
        for column in sweep.columns
        if column not in STUDY_ROW_KEYS
    }
    for column in ("recall", "runtime", "runtime_ms", "dist_comps"):
        if column in aggregations:
            aggregations[column] = "mean"
    grouped = sweep.groupby(STUDY_ROW_KEYS, as_index=False, sort=False, dropna=False)
    collapsed = grouped.agg(aggregations)
    collapsed["measurement_repeats"] = grouped.size()["size"].to_numpy()
    if "qps" in collapsed:
        collapsed["qps"] = 1000.0 / collapsed["runtime_ms"]
    return collapsed


def load_sweep(
    path: Path,
    *,
    require_rho: bool = False,
    strict: bool = False,
    query_type: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = SWEEP_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    df = df[df["algorithm"].isin(PLANS)].copy()
    if query_type is not None:
        if query_type not in supported_query_types():
            raise ValueError(f"unsupported query_type filter {query_type!r}")
        df = df[df["query_type"] == query_type].copy()
    if df.empty:
        raise ValueError(f"{path}: no rows for supported algorithms {sorted(PLANS)}")
    df["plan"] = df["algorithm"].map(PLANS)
    hnsw = df["plan"].isin(["HNSW-pre", "HNSW-post"])
    if "ef_construction" in df.columns:
        ef_construction = pd.to_numeric(df["ef_construction"], errors="coerce")
        if strict and ef_construction[hnsw].isna().any():
            raise ValueError(f"{path}: HNSW rows have missing ef_construction")
        df = df[~hnsw | (ef_construction == 128)].copy()
    elif strict and df.loc[hnsw].duplicated(STUDY_ROW_KEYS, keep=False).any():
        raise ValueError(
            f"{path}: duplicate HNSW actions cannot be disambiguated without ef_construction; "
            "rebuild all_query_results.csv with query_optimizer_analysis.py"
        )
    unknown_tables = sorted(set(df["query_type"].dropna()) - supported_query_types())
    if unknown_tables:
        raise ValueError(f"{path}: unsupported query_type values {unknown_tables}")
    df["N"] = df["query_type"].map(cardinality_for).astype(float)
    df["sigma"] = df["filter_selectivity"].astype(float)
    if strict and df["sigma"].isna().any():
        bad = int(df["sigma"].isna().sum())
        raise ValueError(f"{path}: {bad} supported rows have missing selectivity")
    rho = pd.to_numeric(df["gls_correlation"], errors="coerce")
    rho_values = rho.to_numpy(float)
    invalid_rho = ~np.isfinite(rho_values)
    if require_rho and invalid_rho.any():
        bad = int(invalid_rho.sum())
        raise ValueError(f"{path}: {bad} supported rows have missing/non-finite GLS correlation")
    if strict and ((rho_values < -1.0) | (rho_values > 1.0)).any():
        bad = int(((rho_values < -1.0) | (rho_values > 1.0)).sum())
        raise ValueError(f"{path}: {bad} supported rows have GLS correlation outside [-1, 1]")
    rho = rho.fillna(0.0)
    df["rho"] = np.clip(rho.to_numpy(), -0.999, 0.999)
    for src, dst in zip(GLS_TIER_SOURCE_COLUMNS, RHO_TIER_COLUMNS):
        if src not in df.columns:
            continue
        vals = pd.to_numeric(df[src], errors="coerce")
        if require_rho and vals.isna().any():
            bad = int(vals.isna().sum())
            raise ValueError(f"{path}: {bad} rows have missing {src}")
        vals = vals.fillna(0.0)
        if strict and ((vals < -1.0) | (vals > 1.0)).any():
            bad = int(((vals < -1.0) | (vals > 1.0)).sum())
            raise ValueError(f"{path}: {bad} rows have {src} outside [-1, 1]")
        df[dst] = np.clip(vals.to_numpy(float), -0.999, 0.999)
    df["n_pass"] = np.clip(df["sigma"].to_numpy() * df["N"].to_numpy(), 1.0, None)
    df["runtime_ms"] = df["runtime"].astype(float) * 1000.0
    df = df.dropna(subset=["sigma"])
    df = collapse_repeated_measurements(df)
    if strict:
        validate_study_sweep(df, path)
    return df


def build_pairplan(sweep: pd.DataFrame, recall_target: float = RECALL_TARGET) -> pd.DataFrame:
    """One row per (query, filter, plan): feasibility + best (lowest feasible) hp."""
    group_cols = PAIR_KEYS + ["plan"]
    ordered = sweep.sort_values(group_cols + ["hyperparam"], kind="mergesort")
    first_rows = ordered.drop_duplicates(group_cols, keep="first").copy()
    feas_rows = ordered[ordered["recall"] >= recall_target].drop_duplicates(group_cols, keep="first")
    feas_map = feas_rows.set_index(group_cols)["hyperparam"]

    idx = pd.MultiIndex.from_frame(first_rows[group_cols])
    best_hp_s = feas_map.reindex(idx)
    feasible = best_hp_s.notna().to_numpy(dtype=int)
    best_hp = np.where(
        (first_rows["plan"].to_numpy() != "BF") & (feasible == 1),
        best_hp_s.fillna(0.0).to_numpy(dtype=float),
        0.0,
    )
    out = pd.DataFrame(
        {
            "query_type": first_rows["query_type"].to_numpy(),
            "query_id_num": first_rows["query_id_num"].to_numpy(dtype=int),
            "filter_id": first_rows["filter_id"].to_numpy(dtype=int),
            "k": first_rows["k"].to_numpy(dtype=int),
            "plan": first_rows["plan"].to_numpy(),
            "sigma": first_rows["sigma"].to_numpy(dtype=float),
            "rho": first_rows["rho"].to_numpy(dtype=float),
            "N": first_rows["N"].to_numpy(dtype=float),
            "n_pass": first_rows["n_pass"].to_numpy(dtype=float),
            "feasible": feasible,
            "best_hp": best_hp,
        }
    )
    for col in RHO_TIER_COLUMNS:
        if col in first_rows.columns:
            out[col] = first_rows[col].to_numpy(dtype=float)
    return out


def build_measure(sweep: pd.DataFrame) -> dict:
    """(pair, plan) -> (hps, recall, runtime_ms) sorted ascending (ground truth)."""
    group_cols = PAIR_KEYS + ["plan"]
    ordered = sweep.sort_values(group_cols + ["hyperparam"], kind="mergesort")
    qt_arr = ordered["query_type"].to_numpy()
    qid_arr = ordered["query_id_num"].to_numpy(dtype=int)
    fid_arr = ordered["filter_id"].to_numpy(dtype=int)
    k_arr = ordered["k"].to_numpy(dtype=int)
    plan_arr = ordered["plan"].to_numpy()
    hp_arr = ordered["hyperparam"].to_numpy(dtype=float)
    rec_arr = ordered["recall"].to_numpy(dtype=float)
    rt_arr = ordered["runtime_ms"].to_numpy(dtype=float)

    change = (
        (qt_arr[1:] != qt_arr[:-1])
        | (qid_arr[1:] != qid_arr[:-1])
        | (fid_arr[1:] != fid_arr[:-1])
        | (k_arr[1:] != k_arr[:-1])
        | (plan_arr[1:] != plan_arr[:-1])
    )
    starts = np.concatenate([[0], np.flatnonzero(change) + 1])
    ends = np.concatenate([starts[1:], [len(ordered)]])
    meas = {}
    for s, e in zip(starts, ends):
        meas[(qt_arr[s], int(qid_arr[s]), int(fid_arr[s]), int(k_arr[s]), plan_arr[s])] = (
            hp_arr[s:e],
            rec_arr[s:e],
            rt_arr[s:e],
        )
    return meas


def resolve_measurement(meas, key, target_hp):
    """Effective hp, recall, and runtime at the smallest hp >= target (else max)."""
    hps, rec, rt = meas[key]
    idx = int(np.searchsorted(hps, target_hp, side="left"))
    if idx >= len(hps):
        idx = len(hps) - 1
    return float(hps[idx]), float(rec[idx]), float(rt[idx])


def measure_at(meas, key, target_hp):
    """Measured (recall, runtime_ms) at smallest available hp >= target (else max)."""
    _, rec, rt = resolve_measurement(meas, key, target_hp)
    return rec, rt


def min_feasible(meas, key, recall_target: float = RECALL_TARGET):
    """(runtime_ms, recall) at the lowest available hp reaching recall>=target, else None."""
    hps, rec, rt = meas[key]
    ok = np.where(rec >= recall_target)[0]
    if len(ok) == 0:
        return None
    i = ok[0]
    return float(rt[i]), float(rec[i])


def tuned_baseline(meas, pair, plan, recall_target: float = RECALL_TARGET):
    """Tuned per-plan baseline point for one (query, filter, k) pair.

    Measured (runtime, recall) at the LOWEST swept hp reaching recall >=
    target; if no swept hp reaches it, the max swept hp (best available
    recall). Model-independent: 'always run <plan>, tuned per query'.
    """
    key = (*pair, plan)
    if key not in meas:
        return None
    hps, rec, rt = meas[key]
    ok = np.where(rec >= recall_target)[0]
    i = int(ok[0]) if len(ok) else len(hps) - 1
    return {"rt": float(rt[i]), "rec": float(rec[i]), "hp": float(hps[i]),
            "feasible": bool(len(ok))}


def build_oracle(meas, pairs, recall_target: float = RECALL_TARGET) -> dict:
    """pair -> (oracle_plan, oracle_cost_ms, oracle_recall) over feasible plans."""
    oracle = {}
    for pair in pairs:
        best_plan, best_cost, best_rec = None, np.inf, 1.0
        for plan in ALL_PLANS:
            key = (*pair, plan)
            if key not in meas:
                continue
            if plan == "BF":
                rec, cost = measure_at(meas, key, 0)
            else:
                mf = min_feasible(meas, key, recall_target)
                if mf is None:
                    continue
                cost, rec = mf
            if cost < best_cost:
                best_cost, best_plan, best_rec = cost, plan, rec
        oracle[pair] = (best_plan, best_cost, best_rec)
    return oracle


# --------------------------------------------------------------------------- #
# hp grid snapping (Stage 2)
# --------------------------------------------------------------------------- #
def snap_ceil(pred_hp, grid):
    """Smallest grid value >= pred, clamped to max; below half-min => 0 (infeasible)."""
    if pred_hp < 0.5 * grid[0]:
        return 0
    for v in grid:
        if v >= pred_hp:
            return v
    return grid[-1]


def bump_one(hp, grid):
    if hp <= 0:
        return 0
    i = grid.index(int(hp)) if int(hp) in grid else -1
    return grid[i + 1] if 0 <= i < len(grid) - 1 else hp


def grid_index(hp, grid):
    return grid.index(int(hp)) if int(hp) in grid else -1


# --------------------------------------------------------------------------- #
# per-plan models
# --------------------------------------------------------------------------- #
def _check_feature_set(feature_set: str) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {sorted(FEATURE_SETS)}")


def stage12_feature_columns(feature_set: str = FEATURE_SIGMA_RHO) -> list[str]:
    _check_feature_set(feature_set)
    if feature_set == FEATURE_SIGMA_RHO_TIERS:
        return ["sigma", *RHO_TIER_COLUMNS, "k"]
    if feature_set == FEATURE_SIGMA_RHO:
        return ["sigma", "rho", "k"]
    return ["sigma", "k"]


def stage12_feats(rows: pd.DataFrame, feature_set: str = FEATURE_SIGMA_RHO) -> np.ndarray:
    return rows[stage12_feature_columns(feature_set)].to_numpy()


def correlation_block(rows: pd.DataFrame, feature_set: str = FEATURE_SIGMA_RHO):
    """Stage-3 correlation input: one column, or the three tier columns."""
    _check_feature_set(feature_set)
    if feature_set == FEATURE_SIGMA_RHO_TIERS:
        missing = [col for col in RHO_TIER_COLUMNS if col not in rows.columns]
        if missing:
            raise ValueError(f"tiered GLS features missing columns {missing}")
        return rows[list(RHO_TIER_COLUMNS)].to_numpy(float)
    return rows["rho"].to_numpy()


def estimated_feature_set(columns) -> str:
    """Use the three radii when the estimated sweep carries them."""
    names = set(columns)
    if set(GLS_TIER_SOURCE_COLUMNS) <= names or set(RHO_TIER_COLUMNS) <= names:
        return FEATURE_SIGMA_RHO_TIERS
    return FEATURE_SIGMA_RHO


def stage3_feats(n_pass, rho, N, hp=None, k=None, feature_set: str = FEATURE_SIGMA_RHO):
    _check_feature_set(feature_set)
    cols = [np.log(n_pass)]
    if feature_set == FEATURE_SIGMA_RHO:
        cols.append(np.asarray(rho, float))
    elif feature_set == FEATURE_SIGMA_RHO_TIERS:
        arr = np.asarray(rho, float)
        if arr.ndim != 2 or arr.shape[1] != len(RHO_TIER_COLUMNS):
            raise ValueError(
                "tiered Stage-3 features expect rho with shape (n, 3) "
                "ordered as rho_1, rho_10, rho_100"
            )
        for i in range(arr.shape[1]):
            cols.append(arr[:, i])
    cols.append(np.log(N))
    if hp is not None:
        cols.append(np.log(np.clip(np.asarray(hp, float), 1.0, None)))
    if k is not None:
        cols.append(np.log(np.clip(np.asarray(k, float), 1.0, None)))
    return np.column_stack(cols)


# Backward-compatible name for callers that inspect the default feature set.
STAGE12_FEATS = ["sigma", "rho", "k"]


def fit_models(tr_pp, tr_sweep, min_leaf, grids, feature_set: str = FEATURE_SIGMA_RHO):
    feas_clf, hp_reg, hp_uni, s3 = {}, {}, {}, {}
    for plan in ANN_PLANS:
        tr = tr_pp[tr_pp["plan"] == plan]
        X = stage12_feats(tr, feature_set)
        y_feas = tr["feasible"].to_numpy()
        feas_clf[plan] = CART("clf", min_leaf=min_leaf).fit(X, y_feas)
        hp_uni[plan] = CART("reg", min_leaf=min_leaf).fit(X, tr["best_hp"].to_numpy())
        fr = tr[tr["feasible"] == 1]
        if len(fr) >= 2 * min_leaf:
            hp_reg[plan] = CART("reg", min_leaf=min_leaf).fit(
                stage12_feats(fr, feature_set), np.log(fr["best_hp"].to_numpy()))
        else:  # too few feasible rows -> conservative constant (max grid)
            hp_reg[plan] = float(np.log(grids[plan][-1]))
    for plan in ALL_PLANS:
        s = tr_sweep[tr_sweep["plan"] == plan]
        if len(s) == 0:
            continue
        hp = None if plan == "BF" else s["hyperparam"].to_numpy()
        s3[plan] = LogOLS().fit(
            stage3_feats(s["n_pass"].to_numpy(), correlation_block(s, feature_set),
                         s["N"].to_numpy(), hp, s["k"].to_numpy(), feature_set),
            s["runtime_ms"].to_numpy())
    return feas_clf, hp_reg, hp_uni, s3


def predict_raw_hp(model, X):
    if isinstance(model, float):
        return np.full(len(X), np.exp(model))
    return np.exp(model.predict(X))


def predict_hp(model, X, grid):
    """Ablation hp: exp(log-hp regressor) snapped to grid (>0 always)."""
    raw = predict_raw_hp(model, X)
    return np.array([max(snap_ceil(v, grid), grid[0]) for v in raw])


def snap_by_dataset(raw, query_types, ks, plan, grids, dataset_grids, *, force_positive):
    """Snap each prediction to the action grid supported by its dataset and k."""
    snapped = []
    for value, query_type, k in zip(raw, query_types, ks):
        grid = dataset_grids.get((query_type, int(k), plan), grids[plan])
        hp = snap_ceil(value, grid)
        snapped.append(max(hp, grid[0]) if force_positive else hp)
    return np.asarray(snapped, float)


def _hp_from_quantile(value, grid, sentinel):
    """Grid budget for a pinball prediction, or None when the plan is infeasible."""
    if not np.isfinite(value) or float(value) >= sentinel - 1e-8:
        return None
    return max(snap_ceil(float(np.exp(value)), grid), grid[0])


def run_quantile_policy(
    sweep,
    pp,
    meas,
    oracle,
    grids,
    min_leaf,
    recall_target,
    feature_set,
    deltas=None,
    train_test_splits=None,
    dataset_grids=None,
):
    """Per-plan pinball budget, then the cheapest admissible plan.

    The training label is log of the lowest hyperparameter that reaches the
    recall target, or a sentinel above every grid value when no swept budget
    does. A depth-3 tree minimises pinball loss at level 1 - delta. A sentinel
    prediction drops the plan; every other prediction is snapped upward onto
    the dataset grid. Exact scan is always a candidate. The same grouped folds
    as ``run_cv`` are used (seed 0), unless ``train_test_splits`` provides an
    explicit outer holdout split.
    """
    deltas = list(DELTA_SWEEP if deltas is None else deltas)
    if train_test_splits is None:
        folded = group_folds(pp, 5, 0)
        fold_of = folded[PAIR_KEYS + ["fold"]].drop_duplicates()
        sw = sweep.merge(fold_of, on=PAIR_KEYS, how="left")
        if dataset_grids is None:
            dataset_grids = derive_dataset_grids(sw)
        split_iter = [
            (
                folded[folded["fold"] != fold],
                folded[folded["fold"] == fold],
                sw[sw["fold"] != fold],
            )
            for fold in range(5)
        ]
    else:
        if dataset_grids is None:
            dataset_grids = derive_dataset_grids(sweep)
        split_iter = [
            (tr_pp, te_pp, tr_sw)
            for tr_pp, te_pp, tr_sw, _te_sw in train_test_splits
        ]
    sentinel = budget_sentinel(grids)
    outcomes = {delta: [] for delta in deltas}
    coverage = {
        delta: {plan: [0, 0] for plan in ANN_PLANS}
        for delta in deltas
    }

    for tr_pp, te_pp, tr_sw in split_iter:
        s3 = {}
        for plan in ALL_PLANS:
            sub = tr_sw[tr_sw["plan"] == plan]
            if len(sub) == 0:
                continue
            hp = None if plan == "BF" else sub["hyperparam"].to_numpy()
            s3[plan] = LogOLS().fit(
                stage3_feats(
                    sub["n_pass"].to_numpy(),
                    correlation_block(sub, feature_set),
                    sub["N"].to_numpy(),
                    hp,
                    sub["k"].to_numpy(),
                    feature_set,
                ),
                sub["runtime_ms"].to_numpy(),
            )
        te_bf = te_pp[te_pp["plan"] == "BF"]
        bf_cost = {
            (r.query_type, int(r.query_id_num), int(r.filter_id), int(r.k)): float(cost)
            for r, cost in zip(
                te_bf.itertuples(index=False),
                s3["BF"].predict(stage3_feats(
                    te_bf["n_pass"].to_numpy(),
                    correlation_block(te_bf, feature_set),
                    te_bf["N"].to_numpy(),
                    None,
                    te_bf["k"].to_numpy(),
                    feature_set,
                )),
            )
        }
        for delta in deltas:
            trees = {}
            level = 1.0 - delta
            for plan in ANN_PLANS:
                tr = tr_pp[tr_pp["plan"] == plan]
                y = np.where(
                    tr["feasible"].to_numpy() == 1,
                    np.log(np.clip(tr["best_hp"].to_numpy(), 1.0, None)),
                    sentinel,
                )
                trees[plan] = CART(
                    "quantile", max_depth=3, min_leaf=min_leaf, quantile=level,
                ).fit(stage12_feats(tr, feature_set), y)
            chosen_by_pair = {}
            for plan in ANN_PLANS:
                te = te_pp[te_pp["plan"] == plan]
                if len(te) == 0 or plan not in s3:
                    continue
                values = trees[plan].predict(stage12_feats(te, feature_set))
                query_types = te["query_type"].to_numpy()
                ks = te["k"].to_numpy()
                row_grids = [
                    dataset_grids.get((qt, int(k), plan), grids[plan])
                    for qt, k in zip(query_types, ks)
                ]
                finite_idx = []
                finite_hp = []
                for i, (value, grid) in enumerate(zip(values, row_grids)):
                    hp = _hp_from_quantile(value, grid, sentinel)
                    if hp is None:
                        continue
                    finite_idx.append(i)
                    finite_hp.append(hp)
                costs = {}
                if finite_idx:
                    keep = te.iloc[finite_idx]
                    predicted = s3[plan].predict(stage3_feats(
                        keep["n_pass"].to_numpy(),
                        correlation_block(keep, feature_set),
                        keep["N"].to_numpy(),
                        np.asarray(finite_hp, float),
                        keep["k"].to_numpy(),
                        feature_set,
                    ))
                    for row_i, hp, cost in zip(finite_idx, finite_hp, predicted):
                        costs[row_i] = (hp, float(cost))
                for row_i, r in enumerate(te.itertuples(index=False)):
                    pair = (r.query_type, int(r.query_id_num), int(r.filter_id), int(r.k))
                    grid = row_grids[row_i]
                    if row_i in costs:
                        hp, cost = costs[row_i]
                    else:
                        hp, cost = grid[-1], None
                    _, rec, rt = resolve_measurement(meas, (*pair, plan), hp)
                    if row_i in costs:
                        coverage[delta][plan][0] += 1
                        coverage[delta][plan][1] += int(rec >= recall_target)
                        chosen_by_pair.setdefault(pair, []).append((plan, cost, rec, rt))
                    # Physical-only: this plan at the pinball budget, or its
                    # largest swept budget when the quantile says it cannot hit
                    # the target.
                    chosen_by_pair.setdefault(pair, [])
                    phys = chosen_by_pair.get((pair, "phys"))
                    if phys is None:
                        phys = {}
                        chosen_by_pair[(pair, "phys")] = phys
                    phys[plan] = (rec, rt)

            for r in te_bf.itertuples(index=False):
                pair = (r.query_type, int(r.query_id_num), int(r.filter_id), int(r.k))
                _, bf_rec, bf_rt = resolve_measurement(meas, (*pair, "BF"), 0)
                cands = [("BF", bf_cost[pair], bf_rec, bf_rt)]
                cands.extend(chosen_by_pair.get(pair, []))
                plan, _, rec, rt = min(cands, key=lambda item: item[1])
                op, oc, _ = oracle[pair]
                phys = chosen_by_pair.get((pair, "phys"), {})
                hnsw = phys.get("HNSW-pre", (np.nan, np.nan))
                ivf = phys.get("IVF-pre", (np.nan, np.nan))
                outcomes[delta].append({
                    "plan": plan,
                    "recall": rec,
                    "runtime_ms": rt,
                    "oracle_cost": oc,
                    "is_oracle": plan == op,
                    "bf_rt": bf_rt,
                    "hnsw_rec": hnsw[0],
                    "hnsw_rt": hnsw[1],
                    "ivf_rec": ivf[0],
                    "ivf_rt": ivf[1],
                })

    rows = []
    for delta in deltas:
        got = outcomes[delta]
        recall = np.array([row["recall"] for row in got], float)
        runtime = np.array([row["runtime_ms"] for row in got], float)
        oracle_cost = np.array([row["oracle_cost"] for row in got], float)
        bf = np.array([row["bf_rt"] for row in got], float)
        n = len(got)
        row = {
            "delta": delta,
            "quantile": 1.0 - delta,
            "n": n,
            "avg_recall": float(recall.mean()),
            "recall_fail_rate": float((recall < recall_target).mean()),
            "recall_p10": float(np.percentile(recall, 10)),
            "qps": float(n / (runtime.sum() / 1000.0)),
            "bf_share": float(np.mean([item["plan"] == "BF" for item in got])),
            "chosen_is_oracle_pct": float(100.0 * np.mean([item["is_oracle"] for item in got])),
            "mean_regret": float((runtime / np.clip(oracle_cost, 1e-9, None)).mean()),
            "speedup_vs_bf": float(bf.sum() / runtime.sum()),
        }
        for plan in ("HNSW-pre", "IVF-pre"):
            key = "hnsw" if plan == "HNSW-pre" else "ivf"
            rec = np.array([item[f"{key}_rec"] for item in got], float)
            rt = np.array([item[f"{key}_rt"] for item in got], float)
            ok = np.isfinite(rec) & np.isfinite(rt) & (rt > 0)
            row[f"{key}_qps"] = float(ok.sum() / (rt[ok].sum() / 1000.0)) if ok.any() else np.nan
            row[f"{key}_avg_recall"] = float(rec[ok].mean()) if ok.any() else np.nan
            row[f"{key}_fail_rate"] = float((rec[ok] < recall_target).mean()) if ok.any() else np.nan
        for plan in ANN_PLANS:
            n_pred, n_hit = coverage[delta][plan]
            row[f"cov_{plan}"] = n_hit / n_pred if n_pred else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# cross-validated evaluation
# --------------------------------------------------------------------------- #
def group_folds(pp, n_folds=5, seed=0):
    group_keys = ["query_type", "query_id_num"]
    groups = (
        pp[group_keys]
        .drop_duplicates()
        .sort_values(group_keys, kind="mergesort")
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    groups = groups.iloc[rng.permutation(len(groups))].reset_index(drop=True)
    groups["fold"] = np.arange(len(groups)) % n_folds
    return pp.merge(groups, on=group_keys, how="left")


def run_cv(
    sweep,
    pp,
    meas,
    oracle,
    grids,
    n_folds=5,
    seed=0,
    min_leaf=30,
    recall_target: float = RECALL_TARGET,
    feature_set: str = FEATURE_SIGMA_RHO,
    train_test_splits=None,
    dataset_grids=None,
):
    if train_test_splits is None:
        pp = group_folds(pp, n_folds, seed)
        fold_of = pp[PAIR_KEYS + ["fold"]].drop_duplicates()
        sweep = sweep.merge(fold_of, on=PAIR_KEYS, how="left")
        if dataset_grids is None:
            dataset_grids = derive_dataset_grids(sweep)
        split_iter = [
            (
                pp[pp["fold"] != fold],
                pp[pp["fold"] == fold],
                sweep[sweep["fold"] != fold],
                sweep[sweep["fold"] == fold],
            )
            for fold in range(n_folds)
        ]
    else:
        if dataset_grids is None:
            dataset_grids = derive_dataset_grids(sweep)
        split_iter = list(train_test_splits)

    stage1 = {p: {"acc": [], "auc": []} for p in ANN_PLANS}
    stage2 = {p: {"hit": 0, "n": 0, "step": []} for p in ANN_PLANS}
    stage2u = {p: {"acc": [], "hit": 0, "n": 0} for p in ANN_PLANS}  # unified variant
    s3_true = {p: {"yt": [], "yp": []} for p in ALL_PLANS}
    s3_pred = {p: {"yt": [], "yp": []} for p in ANN_PLANS}
    records = []

    for tr_pp, te_pp, tr_sweep, te_sweep in split_iter:
        feas_clf, hp_reg, hp_uni, s3 = fit_models(
            tr_pp, tr_sweep, min_leaf, grids, feature_set
        )

        # Stage 3 "true hp": predict every test sweep row
        for plan in ALL_PLANS:
            sub = te_sweep[te_sweep["plan"] == plan]
            if len(sub) == 0 or plan not in s3:
                continue
            hp = None if plan == "BF" else sub["hyperparam"].to_numpy()
            pred = s3[plan].predict(
                stage3_feats(sub["n_pass"].to_numpy(), correlation_block(sub, feature_set),
                             sub["N"].to_numpy(), hp, sub["k"].to_numpy(), feature_set))
            s3_true[plan]["yt"].append(sub["runtime_ms"].to_numpy())
            s3_true[plan]["yp"].append(pred)

        # per-plan predictions on the test pairs (vectorised per plan)
        te_ann = te_pp[te_pp["plan"].isin(ANN_PLANS)].copy()
        cols = {"P": np.zeros(len(te_ann)), "hp": np.zeros(len(te_ann)),
                "hpb": np.zeros(len(te_ann)), "cost": np.zeros(len(te_ann)),
                "costb": np.zeros(len(te_ann)), "rt": np.zeros(len(te_ann)),
                "rtb": np.zeros(len(te_ann)), "rec": np.zeros(len(te_ann)),
                "recb": np.zeros(len(te_ann)), "effective_hp": np.zeros(len(te_ann)),
                "effective_hpb": np.zeros(len(te_ann)), "hp_uni": np.zeros(len(te_ann))}
        plan_arr = te_ann["plan"].to_numpy()
        for plan in ANN_PLANS:
            m = plan_arr == plan
            if not m.any():
                continue
            grid = grids[plan]
            sub = te_ann[m]
            X = stage12_feats(sub, feature_set)
            query_types = sub["query_type"].to_numpy()
            n_pass = sub["n_pass"].to_numpy()
            rho = correlation_block(sub, feature_set)
            N = sub["N"].to_numpy()
            kk = sub["k"].to_numpy()
            row_grids = [
                dataset_grids.get((qt, int(k), plan), grid)
                for qt, k in zip(query_types, kk)
            ]
            P = feas_clf[plan].predict(X)
            hp = snap_by_dataset(
                predict_raw_hp(hp_reg[plan], X),
                query_types,
                kk,
                plan,
                grids,
                dataset_grids,
                force_positive=True,
            )
            hpb = np.array([bump_one(v, row_grid) for v, row_grid in zip(hp, row_grids)])
            cost = s3[plan].predict(stage3_feats(n_pass, rho, N, hp, kk, feature_set))
            costb = s3[plan].predict(stage3_feats(n_pass, rho, N, hpb, kk, feature_set))
            keys = list(zip(sub["query_type"], sub["query_id_num"].astype(int),
                            sub["filter_id"].astype(int), sub["k"].astype(int)))
            rr = np.array([resolve_measurement(meas, (*key, plan), h) for key, h in zip(keys, hp)])
            rrb = np.array(
                [resolve_measurement(meas, (*key, plan), h) for key, h in zip(keys, hpb)]
            )
            uni = snap_by_dataset(
                hp_uni[plan].predict(X),
                query_types,
                kk,
                plan,
                grids,
                dataset_grids,
                force_positive=False,
            )
            cols["P"][m], cols["hp"][m], cols["hpb"][m] = P, hp, hpb
            cols["cost"][m], cols["costb"][m] = cost, costb
            cols["effective_hp"][m], cols["rec"][m], cols["rt"][m] = (
                rr[:, 0], rr[:, 1], rr[:, 2]
            )
            cols["effective_hpb"][m], cols["recb"][m], cols["rtb"][m] = (
                rrb[:, 0], rrb[:, 1], rrb[:, 2]
            )
            cols["hp_uni"][m] = uni

            # --- Stage 1/2 diagnostics -----------------------------------
            feas_true = sub["feasible"].to_numpy()
            stage1[plan]["acc"].append(float(((P >= 0.5).astype(int) == feas_true).mean()))
            stage1[plan]["auc"].append(auc(feas_true, P))
            gated = P >= 0.5
            stage2[plan]["n"] += int(gated.sum())
            stage2[plan]["hit"] += int((rr[gated, 1] >= recall_target).sum())
            gated_grids = [row_grid for row_grid, keep in zip(row_grids, gated) if keep]
            for h, bt, ft, row_grid in zip(
                hp[gated],
                sub["best_hp"].to_numpy()[gated],
                feas_true[gated],
                gated_grids,
            ):
                if ft == 1:
                    ti, pi = grid_index(bt, row_grid), grid_index(h, row_grid)
                    if ti >= 0 and pi >= 0:
                        stage2[plan]["step"].append(pi - ti)
            # Stage 3 at predicted (ablation) hp on gated pairs
            s3_pred[plan]["yt"].extend(rr[gated, 2].tolist())
            s3_pred[plan]["yp"].extend(cost[gated].tolist())
            # unified-model feasibility/hit (reference)
            feas_uni = uni > 0
            stage2u[plan]["acc"].append(float((feas_uni.astype(int) == feas_true).mean()))
            stage2u[plan]["n"] += int(feas_uni.sum())
            if feas_uni.any():
                ru = np.array([measure_at(meas, (*key, plan), h)[0]
                               for key, h, fu in zip(keys, uni, feas_uni) if fu])
                stage2u[plan]["hit"] += int((ru >= recall_target).sum())

        for c in cols:
            te_ann[c] = cols[c]

        # BF predicted cost per test pair (batch)
        te_bf = te_pp[te_pp["plan"] == "BF"].copy()
        te_bf["cost"] = s3["BF"].predict(
            stage3_feats(te_bf["n_pass"].to_numpy(), correlation_block(te_bf, feature_set),
                         te_bf["N"].to_numpy(), None, te_bf["k"].to_numpy(), feature_set))
        bf_cost = {(r.query_type, int(r.query_id_num), int(r.filter_id), int(r.k)): float(r.cost)
                   for r in te_bf.itertuples()}

        # assemble per-pair optimizer records
        by_pair: dict[tuple, list] = {}
        for r in te_ann.itertuples(index=False):
            pair = (r.query_type, int(r.query_id_num), int(r.filter_id), int(r.k))
            by_pair.setdefault(pair, []).append(r)

        for r_bf in te_bf.itertuples(index=False):
            pair = (r_bf.query_type, int(r_bf.query_id_num), int(r_bf.filter_id), int(r_bf.k))
            sub_rows = by_pair.get(pair, [])
            bf_key = (*pair, "BF")
            bf_hp, bf_rec, bf_rt = resolve_measurement(meas, bf_key, 0)
            cands = {"BF": {"p_feas": 1.0, "cost": bf_cost[pair], "cost_b": bf_cost[pair],
                            "hp": 0.0, "hp_b": 0.0,
                            "effective_hp": bf_hp, "effective_hp_b": bf_hp,
                            "rt": bf_rt, "rt_b": bf_rt, "rec": bf_rec, "rec_b": bf_rec}}
            phys = {}
            for r in sub_rows:
                if r.P >= 0.5:
                    cands[r.plan] = {"p_feas": float(r.P), "cost": r.cost, "cost_b": r.costb,
                                     "hp": float(r.hp), "hp_b": float(r.hpb),
                                     "effective_hp": float(r.effective_hp),
                                     "effective_hp_b": float(r.effective_hpb),
                                     "rt": r.rt, "rt_b": r.rtb, "rec": r.rec, "rec_b": r.recb}
                phys[r.plan] = {
                    "p_feas": float(r.P),
                    "hp": float(r.hp),
                    "effective_hp": float(r.effective_hp),
                    "rt": float(r.rt),
                    "rec": float(r.rec),
                }
            bf_point = {"rt": bf_rt, "rec": bf_rec, "hp": 0.0, "feasible": True}
            hnsw = tuned_baseline(meas, pair, "HNSW-pre", recall_target) or dict(bf_point)
            ivf = tuned_baseline(meas, pair, "IVF-pre", recall_target) or dict(bf_point)
            op, oc, orc = oracle[pair]
            records.append({"pair": pair, "cands": cands, "oracle_plan": op,
                            "oracle_cost": oc, "oracle_recall": orc,
                            "bf_rt": bf_rt, "hnsw": hnsw, "ivf": ivf,
                            "phys": phys})

    diag = {"stage1": stage1, "stage2": stage2, "stage2u": stage2u,
            "s3_true": s3_true, "s3_pred": s3_pred}
    return records, diag


# --------------------------------------------------------------------------- #
# decision rule + optimizer scoring (cheap; sweep gamma/lambda/bump w/o refit)
# --------------------------------------------------------------------------- #
def decision_details(rec, gamma, lam, bump):
    ck, rk = ("cost_b", "rt_b") if bump else ("cost", "rt")
    reck = "rec_b" if bump else "rec"
    hp_key = "hp_b" if bump else "hp"
    effective_hp_key = "effective_hp_b" if bump else "effective_hp"
    best_plan, best_score = "BF", np.inf
    for plan, c in rec["cands"].items():
        m = 1.0 if plan == "BF" else gamma * (1.0 + lam * (1.0 - c["p_feas"]))
        score = c[ck] * m
        if score < best_score:
            best_score, best_plan = score, plan
    c = rec["cands"][best_plan]
    return {
        "plan": best_plan,
        "requested_hyperparam": float(c[hp_key]),
        "effective_hyperparam": float(c[effective_hp_key]),
        "predicted_cost_ms": float(c[ck]),
        "runtime_ms": float(c[rk]),
        "recall": float(c[reck]),
        "p_feasible": float(c["p_feas"]),
    }


def decide(rec, gamma, lam, bump):
    """Backward-compatible compact decision result."""
    d = decision_details(rec, gamma, lam, bump)
    return d["plan"], d["runtime_ms"], d["recall"]


ROBUSTNESS_BASE_THRESHOLDS = [0.90, 0.80, 0.50]


def robustness_thresholds(recall_target: float) -> list[float]:
    """Keep 0.90 / 0.80 / 0.50; always include the active target."""
    out = [float(recall_target)]
    for t in ROBUSTNESS_BASE_THRESHOLDS:
        if abs(t - recall_target) > 1e-12:
            out.append(t)
    return out


def _bad_rate_key(thr: float, recall_target: float) -> str:
    if abs(thr - recall_target) < 1e-12:
        return "recall_fail_rate"
    return f"bad_recall_rate_{round(thr * 100):03d}"


def optimizer_metrics(records, gamma, lam, bump=False, recall_target: float = RECALL_TARGET):
    n = len(records)
    realised = np.empty(n)
    oracle = np.empty(n)
    bf = np.empty(n)
    hnsw = np.empty(n)
    ivf = np.empty(n)
    recall = np.empty(n)
    requested_hp = np.empty(n)
    effective_hp = np.empty(n)
    predicted_cost = np.empty(n)
    p_feasible = np.empty(n)
    is_oracle = np.zeros(n, bool)
    chosen = []
    for i, rec in enumerate(records):
        decision = decision_details(rec, gamma, lam, bump)
        plan = decision["plan"]
        realised[i] = decision["runtime_ms"]
        recall[i] = decision["recall"]
        requested_hp[i] = decision["requested_hyperparam"]
        effective_hp[i] = decision["effective_hyperparam"]
        predicted_cost[i] = decision["predicted_cost_ms"]
        p_feasible[i] = decision["p_feasible"]
        oracle[i] = rec["oracle_cost"]
        bf[i] = rec["bf_rt"]
        # tuned per-query baselines (independent of the bump knob)
        hnsw[i] = rec["hnsw"]["rt"]
        ivf[i] = rec["ivf"]["rt"]
        is_oracle[i] = plan == rec["oracle_plan"]
        chosen.append(plan)
    fail = recall < recall_target
    hnsw_rec = np.array([rec["hnsw"]["rec"] for rec in records])
    ivf_rec = np.array([rec["ivf"]["rec"] for rec in records])
    bf_rec = np.array([rec["cands"]["BF"]["rec"] for rec in records])
    regret = realised / np.clip(oracle, 1e-9, None)
    tot_s = realised.sum() / 1000.0
    out = {
        "gamma": gamma, "lam": lam, "bump": int(bump), "n": n,
        "avg_recall": float(recall.mean()),
        "recall_fail_rate": float(fail.mean()),
        # robustness: tail of the realised-recall distribution
        "recall_p10": float(np.percentile(recall, 10)),
        "bad_recall_rate_090": float((recall < 0.90).mean()),
        "bad_recall_rate_080": float((recall < 0.80).mean()),
        "bad_recall_rate_050": float((recall < 0.50).mean()),
        "qps": float(n / tot_s),
        "chosen_is_oracle_pct": float(100 * is_oracle.mean()),
        "mean_regret": float(regret.mean()), "median_regret": float(np.median(regret)),
        "total_realised_ms": float(realised.sum()), "total_oracle_ms": float(oracle.sum()),
        "total_bf_ms": float(bf.sum()), "total_hnsw_ms": float(hnsw.sum()),
        "total_ivf_ms": float(ivf.sum()),
        "bf_qps": float(n / (bf.sum() / 1000.0)),
        "bf_avg_recall": float(bf_rec.mean()),
        "oracle_qps": float(n / (oracle.sum() / 1000.0)),
        "hnsw_qps": float(n / (hnsw.sum() / 1000.0)),
        "hnsw_avg_recall": float(hnsw_rec.mean()),
        "hnsw_fail_rate": float((hnsw_rec < recall_target).mean()),
        "ivf_qps": float(n / (ivf.sum() / 1000.0)),
        "ivf_avg_recall": float(ivf_rec.mean()),
        "ivf_fail_rate": float((ivf_rec < recall_target).mean()),
        "speedup_vs_bf": float(bf.sum() / realised.sum()),
        "speedup_vs_hnsw": float(hnsw.sum() / realised.sum()),
        "speedup_vs_ivf": float(ivf.sum() / realised.sum()),
        "chosen": chosen,
        "chosen_hyperparam": requested_hp,
        "effective_hyperparam": effective_hp,
        "chosen_predicted_cost_ms": predicted_cost,
        "chosen_p_feasible": p_feasible,
        "realised": realised,
        "recall": recall,
        "fail": fail,
    }
    extra_named = f"bad_recall_rate_{round(float(recall_target) * 100):03d}"
    if extra_named not in out:
        out[extra_named] = float((recall < recall_target).mean())
    return out


def static_plan_curve(meas, pairs, plan: str) -> pd.DataFrame:
    """QPS–recall curve of one logical plan with a single hyperparameter for every query.

    Each point fixes the swept search parameter (efSearch or nprobe) to one
    value on every pair and reports mean recall and aggregate throughput
    ``n / sum(runtime_seconds)``. Only hyperparameter values present for every
    pair are included, so each point is the same policy on the full workload.
    """
    series = []
    for pair in pairs:
        key = (*pair, plan)
        if key not in meas:
            raise ValueError(f"missing {plan} measurements for pair {pair}")
        series.append(meas[key])
    def _hp_keys(values):
        return {int(round(float(v))) for v in values}

    common = _hp_keys(series[0][0])
    for hps, _rec, _rt in series[1:]:
        common &= _hp_keys(hps)
    if not common:
        raise ValueError(f"{plan}: no hyperparameter is swept for every query")

    rows = []
    for hp in sorted(common):
        recalls = np.empty(len(series))
        runtimes = np.empty(len(series))
        for i, (hps, rec, rt) in enumerate(series):
            idx = int(np.searchsorted(hps, hp, side="left"))
            recalls[i] = rec[idx]
            runtimes[i] = rt[idx]
        total_s = float(runtimes.sum()) / 1000.0
        rows.append({
            "kind": "curve",
            "name": plan,
            "plan": plan,
            "hyperparam": float(hp),
            "hp_name": PHYSICAL_PLAN_HP.get(plan, "hyperparam"),
            "avg_recall": float(recalls.mean()),
            "qps": float(len(series) / total_s) if total_s > 0 else np.nan,
            "n": len(series),
        })
    return pd.DataFrame(rows)


def physical_optimizer_metrics(records, plan: str, recall_target: float = RECALL_TARGET) -> dict:
    """Learned search-parameter choice with the logical plan held fixed.

    Uses the same cross-validated hp regressor as the full optimizer (lowest
    swept hp predicted to reach the recall target, snapped to the grid). The
    feasibility gate and the gamma/lambda plan router are not applied: every
    query runs ``plan`` at the predicted efSearch or nprobe.
    """
    if plan not in PHYSICAL_PLAN_HP:
        raise ValueError(f"unsupported physical plan {plan!r}")
    n = len(records)
    realised = np.empty(n)
    recall = np.empty(n)
    hp = np.empty(n)
    for i, rec in enumerate(records):
        pred = rec.get("phys", {}).get(plan)
        if pred is None:
            raise ValueError(f"missing physical prediction for {plan} on pair {rec['pair']}")
        realised[i] = pred["rt"]
        recall[i] = pred["rec"]
        hp[i] = pred["hp"]
    total_s = float(realised.sum()) / 1000.0
    return {
        "plan": plan,
        "hp_name": PHYSICAL_PLAN_HP[plan],
        "n": n,
        "avg_recall": float(recall.mean()),
        "qps": float(n / total_s) if total_s > 0 else np.nan,
        "recall_fail_rate": float((recall < recall_target).mean()),
        "hyperparam": hp,
        "recall": recall,
        "realised": realised,
    }


def _aggregate_qps(runtimes_ms) -> float:
    """Workload throughput: query count divided by total search time."""
    runtimes_ms = np.asarray(runtimes_ms, float)
    total_s = float(runtimes_ms.sum()) / 1000.0
    if total_s <= 0:
        return np.nan
    return float(len(runtimes_ms) / total_s)


def _qps_recall_scatter_points(point_rows: list[dict], gamma: float) -> list[dict]:
    """Legend entries for the single-point policies on the QPS–recall plot."""
    style = {
        "always-BF": ("s", "#e2585f", 120, 4),
        "oracle": ("D", "#39c07a", 130, 5),
        "optimizer": ("o", "#5aa0ff", 170, 6),
        "physical-HNSW-pre": ("X", "#c47b12", 170, 6),
        "physical-IVF-pre": ("P", "#1d6f66", 180, 6),
    }

    def label_for(row):
        qps = row["qps"]
        rec = row["avg_recall"]
        name = row["name"]
        if name == "always-BF":
            return f"always-BF (recall {rec:.3f})"
        if name == "oracle":
            return f"oracle ({qps:.0f} QPS, recall {rec:.3f})"
        if name == "optimizer":
            return f"query optimizer (g={gamma:g})"
        if name == "physical-HNSW-pre":
            return "optimizer, HNSW only (efSearch)"
        if name == "physical-IVF-pre":
            return "optimizer, IVF only (nprobe)"
        return name

    # Offsets keep the clustered optimizer / oracle / IVF-only labels apart.
    offsets = {
        "always-BF": (-36, 12),
        "oracle": (10, 8),
        "optimizer": (-52, 12),
        "physical-HNSW-pre": (8, -16),
        "physical-IVF-pre": (-6, -18),
    }
    points = []
    for row in point_rows:
        marker, color, size, zorder = style[row["name"]]
        points.append({
            "avg_recall": row["avg_recall"],
            "qps": row["qps"],
            "marker": marker,
            "color": color,
            "s": size,
            "zorder": zorder,
            "label": label_for(row),
            "annotate": f"{row['qps']:.0f} QPS",
            "xytext": offsets[row["name"]],
        })
    return points


def plot_qps_recall_plans(
    curves: pd.DataFrame,
    points: list[dict],
    out_png,
    label: str,
    recall_target: float = RECALL_TARGET,
    k10_footnote: bool = False,
):
    """Throughput vs average recall: static HNSW/IVF curves plus single policy points.

    Curves sweep one global efSearch or nprobe. Brute force, the per-query
    oracle, the full query optimizer, and the physical-only optimizers are
    single points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for plan in PHYSICAL_PLANS:
        sub = curves[curves["plan"] == plan].sort_values("hyperparam")
        if sub.empty:
            continue
        style = PHYSICAL_CURVE_STYLE[plan]
        hp_name = PHYSICAL_PLAN_HP[plan]
        ax.plot(
            sub["avg_recall"],
            sub["qps"],
            linestyle="-",
            marker=style["marker"],
            color=style["color"],
            ms=5,
            lw=1.8,
            label=f"{plan} curve (fixed {hp_name})",
            zorder=3,
        )
    for point in points:
        ax.scatter(
            [point["avg_recall"]],
            [point["qps"]],
            marker=point["marker"],
            s=point.get("s", 130),
            color=point["color"],
            edgecolors="black",
            linewidths=0.6,
            zorder=point.get("zorder", 5),
            label=point["label"],
        )
        if point.get("annotate"):
            ax.annotate(
                point["annotate"],
                (point["avg_recall"], point["qps"]),
                textcoords="offset points",
                xytext=point.get("xytext", (6, 6)),
                fontsize=7.5,
                color="#222",
            )
    ax.axvline(recall_target, color="#888", ls="--", lw=1, alpha=0.7,
               label=f"recall target {format_recall_target(recall_target)}")
    ax.set_yscale("log")
    ax.set_xlabel("Average recall of chosen plans")
    ax.set_ylabel("Search-only throughput [QPS, log]")
    rt = format_recall_target(recall_target)
    ax.set_title(f"Throughput vs average recall -- {label} (recall ≥ {rt})")
    ax.grid(True, alpha=0.3)
    ax.legend(
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
    )
    footnote = (
        "Curves: every query uses the same swept hyperparameter. "
        "Brute force is exact (recall 1.0). "
        "Oracle: per query, the fastest plan among those reaching the recall target, "
        "each ANN plan at its cheapest swept hp; BF is always a candidate. "
        "Query optimizer chooses both the logical plan and the search hp. "
        "Physical-only optimizers keep the logical plan fixed (HNSW-pre or IVF-pre / IVFFlat) "
        "and only predict efSearch or nprobe, using the same cross-validated hp model "
        "as the query optimizer (no feasibility gate, no gamma margin). "
        "Optimizer decision time is not charged."
    )
    if k10_footnote:
        footnote = footnote + " " + K10_RECALL_FOOTNOTE
    fig.text(0.01, -0.02, footnote, fontsize=7, ha="left", va="top", wrap=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved QPS-recall plan plot -> {out_png}")


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def report_stages(diag):
    s1, s2, s2u = diag["stage1"], diag["stage2"], diag["stage2u"]
    print("\n--- Stage 1 (feasibility classifier) + Stage 2 (hp) ---")
    print(f"  {'plan':10s} {'feas_acc':>8s} {'feas_auc':>8s} {'hp_hit':>7s} "
          f"{'n_run':>6s} {'step_bias':>9s} | {'uni_acc':>7s} {'uni_hit':>7s}")
    rows = []
    for p in ANN_PLANS:
        acc_values = np.asarray(s1[p]["acc"], float)
        auc_values = np.asarray(s1[p]["auc"], float)
        acc = float(acc_values[np.isfinite(acc_values)].mean()) if np.isfinite(acc_values).any() else np.nan
        au = float(auc_values[np.isfinite(auc_values)].mean()) if np.isfinite(auc_values).any() else np.nan
        hit = s2[p]["hit"] / s2[p]["n"] if s2[p]["n"] else np.nan
        bias = np.mean(s2[p]["step"]) if s2[p]["step"] else np.nan
        uacc = np.nanmean(s2u[p]["acc"]) if s2u[p]["acc"] else np.nan
        uhit = s2u[p]["hit"] / s2u[p]["n"] if s2u[p]["n"] else np.nan
        print(f"  {p:10s} {acc:8.3f} {au:8.3f} {hit:7.3f} {s2[p]['n']:6d} "
              f"{bias:9.2f} | {uacc:7.3f} {uhit:7.3f}")
        rows.append({"plan": p, "feas_acc": acc, "feas_auc": au, "hp_hit_rate": hit,
                     "n_predicted_feasible": s2[p]["n"], "hp_step_bias": bias,
                     "hp_step_mae": float(np.mean(np.abs(s2[p]["step"]))) if s2[p]["step"] else np.nan,
                     "unified_feas_acc": uacc, "unified_hp_hit_rate": uhit})
    return rows


def report_s3(diag):
    st, sp = diag["s3_true"], diag["s3_pred"]
    print("\n--- Stage 3 (runtime log-R^2) ---")
    print(f"  {'plan':10s} {'@true_hp':>9s} {'@pred_hp':>9s}")
    rows = []
    for p in ALL_PLANS:
        yt = np.concatenate(st[p]["yt"]) if st[p]["yt"] else np.array([])
        yp = np.concatenate(st[p]["yp"]) if st[p]["yp"] else np.array([])
        r2t = log_r2(yt, yp) if len(yt) else np.nan
        r2p = log_r2(np.array(sp[p]["yt"]), np.array(sp[p]["yp"])) if p in sp and sp[p]["yt"] else np.nan
        print(f"  {p:10s} {r2t:9.3f} {r2p:9.3f}")
        rows.append({"plan": p, "logR2_true_hp": r2t, "logR2_pred_hp": r2p})
    return rows


def report_optimizer(m, label, recall_target: float = RECALL_TARGET):
    print(f"\n--- Optimizer [{label}]  (gamma={m['gamma']}, lambda={m['lam']}, "
          f"safety_bump={bool(m['bump'])}) ---")
    print(f"  AVERAGE recall of chosen plans : {m['avg_recall']:.4f}   "
          f"(recall<{format_recall_target(recall_target)} on {100*m['recall_fail_rate']:.1f}% of pairs)")
    print(f"  ROBUSTNESS (recall tail)       : p10={m['recall_p10']:.3f}   "
          f"<0.9 on {100*m['bad_recall_rate_090']:.1f}%, "
          f"<0.8 on {100*m['bad_recall_rate_080']:.1f}%, "
          f"<0.5 on {100*m['bad_recall_rate_050']:.1f}% of pairs")
    print(f"  SEARCH-ONLY THROUGHPUT         : {m['qps']:.1f} QPS  "
          f"(BF {m['bf_qps']:.1f}, oracle {m['oracle_qps']:.1f}, "
          f"HNSW-pre tuned {m['hnsw_qps']:.1f}, IVF-pre tuned {m['ivf_qps']:.1f})")
    print(f"  chosen == oracle plan          : {m['chosen_is_oracle_pct']:.1f}%")
    print(f"  mean regret (realised/oracle)  : {m['mean_regret']:.3f}   median {m['median_regret']:.3f}")
    print(f"  vs always-BF                   : {m['speedup_vs_bf']:.2f}x throughput  "
          f"(BF measured recall {m['bf_avg_recall']:.3f})")
    print(f"  vs always-HNSW-pre (tuned)     : {m['speedup_vs_hnsw']:.2f}x throughput  "
          f"(HNSW-pre avg recall {m['hnsw_avg_recall']:.4f})")
    print(f"  vs always-IVF-pre (tuned)      : {m['speedup_vs_ivf']:.2f}x throughput  "
          f"(IVF-pre avg recall {m['ivf_avg_recall']:.4f})")
    mix = pd.Series(m["chosen"]).value_counts()
    print("  chosen-plan mix: " + ", ".join(f"{p}={c}" for p, c in mix.items()))


def build_selection_frame(records, metrics, feat, bump_metrics, variant, feature_set):
    """Per-pair audit trail for the default and one-step safety-bump policies."""
    return pd.DataFrame({
        "variant": variant,
        "feature_set": feature_set,
        "query_type": [r["pair"][0] for r in records],
        "query_id_num": [r["pair"][1] for r in records],
        "filter_id": [r["pair"][2] for r in records],
        "k": [r["pair"][3] for r in records],
        "sigma": [feat[r["pair"]][0] for r in records],
        "rho": [feat[r["pair"]][1] for r in records],
        "chosen": metrics["chosen"],
        "chosen_hyperparam": metrics["chosen_hyperparam"],
        "effective_hyperparam": metrics["effective_hyperparam"],
        "chosen_predicted_cost_ms": metrics["chosen_predicted_cost_ms"],
        "chosen_p_feasible": metrics["chosen_p_feasible"],
        "realised_ms": metrics["realised"],
        "recall": metrics["recall"],
        "recall_fail": metrics["fail"],
        "safety_bump_chosen": bump_metrics["chosen"],
        "safety_bump_hyperparam": bump_metrics["chosen_hyperparam"],
        "safety_bump_effective_hyperparam": bump_metrics["effective_hyperparam"],
        "safety_bump_realised_ms": bump_metrics["realised"],
        "safety_bump_recall": bump_metrics["recall"],
        "oracle_plan": [r["oracle_plan"] for r in records],
        "oracle_ms": [r["oracle_cost"] for r in records],
        "bf_ms": [r["bf_rt"] for r in records],
    })


def _label_rows(pareto, max_labels=10):
    """Indices of ~max_labels evenly spaced sweep points (incl. endpoints)."""
    n = len(pareto)
    step = max(1, int(np.ceil((n - 1) / max(1, max_labels - 1))))
    idx = set(range(0, n, step))
    idx.add(n - 1)
    return idx


def plot_pareto(pareto, out_png, label, recall_target: float = RECALL_TARGET, k10_footnote: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frontier = pareto_envelope_frame(pareto)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(frontier["avg_recall"], frontier["qps"], "o-", color="#5aa0ff", ms=4)
    labelled = _label_rows(frontier)
    for i, (_, r) in enumerate(frontier.iterrows()):
        if i not in labelled:
            continue
        ax.annotate(f"g={r['gamma']:g}", (r["avg_recall"], r["qps"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    rt = format_recall_target(recall_target)
    ax.axvline(recall_target, color="#e2585f", ls="--", lw=1, alpha=0.7,
               label=f"recall target {rt}")
    ax.set_yscale("log")
    ax.set_xlabel("Average recall of chosen plans")
    ax.set_ylabel("Search-only throughput [QPS, log]")
    ax.set_title(
        f"Non-dominated gamma-sweep envelope -- {label} (recall target {rt})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    apply_k10_footnote(fig, k10_footnote)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved Pareto plot -> {out_png}")


def pareto_envelope_frame(
    pareto: pd.DataFrame, quality_col: str = "avg_recall"
) -> pd.DataFrame:
    """Non-dominated points where ``quality_col`` and QPS are both maximized."""
    if pareto.empty:
        raise ValueError("cannot build a Pareto envelope from an empty frame")
    missing = {quality_col, "qps"} - set(pareto.columns)
    if missing:
        raise ValueError(f"cannot build Pareto envelope; missing columns {sorted(missing)}")
    ordered = pareto.sort_values([quality_col, "qps"], kind="mergesort")
    envelope = []
    best_qps = -np.inf
    for _, row in ordered.iloc[::-1].iterrows():
        qps = float(row["qps"])
        if qps > best_qps:
            envelope.append(row)
            best_qps = qps
    return pd.DataFrame(envelope[::-1]).reset_index(drop=True)


def safety_pareto_envelope_frame(points: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated points maximizing target success rate and search-only QPS."""
    if "recall_fail_rate" not in points:
        raise ValueError("safety Pareto requires recall_fail_rate")
    with_success = points.copy()
    with_success["recall_success_rate"] = 1.0 - with_success["recall_fail_rate"]
    return (
        pareto_envelope_frame(with_success, quality_col="recall_success_rate")
        .sort_values("recall_fail_rate", kind="mergesort")
        .reset_index(drop=True)
    )


def plot_safety_pareto(
    points,
    out_png,
    label,
    recall_target: float = RECALL_TARGET,
    k10_footnote: bool = False,
):
    """Plot non-dominated search-only QPS versus per-query target-failure rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frontier = safety_pareto_envelope_frame(points)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        100.0 * frontier["recall_fail_rate"],
        frontier["qps"],
        "o-",
        color="#5aa0ff",
        ms=4,
    )
    labelled = _label_rows(frontier)
    for i, (_, row) in enumerate(frontier.iterrows()):
        if i not in labelled:
            continue
        ax.annotate(
            f"g={row['gamma']:g}",
            (100.0 * row["recall_fail_rate"], row["qps"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    rt = format_recall_target(recall_target)
    ax.set_yscale("log")
    ax.set_xlabel(f"Queries with Recall@k < {rt} [%] (lower is safer)")
    ax.set_ylabel("Search-only throughput [QPS, log]")
    ax.set_title(f"Recall-safety Pareto envelope -- {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    apply_k10_footnote(fig, k10_footnote)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved recall-safety Pareto plot -> {out_png}")


def interpolate_frontier_point(frontier: pd.DataFrame, recall_level: float):
    """Realizable random-policy mixture between adjacent frontier points.

    Recall and mean latency mix linearly. QPS therefore uses harmonic, not
    geometric, interpolation. ``mix_high`` is the share of requests assigned
    to the higher-recall bracketing policy.
    """
    points = frontier.sort_values("avg_recall")
    xs = points["avg_recall"].to_numpy(float)
    ys = points["qps"].to_numpy(float)
    gammas = points["gamma"].to_numpy(float)
    if recall_level < xs[0] - 1e-9 or recall_level > xs[-1] + 1e-9:
        return None
    if len(xs) == 1:
        if abs(recall_level - xs[0]) > 1e-9:
            return None
        return {
            "qps": float(ys[0]),
            "gamma_low": float(gammas[0]),
            "gamma_high": float(gammas[0]),
            "mix_high": 0.0,
        }
    for i in range(len(xs) - 1):
        if xs[i] - 1e-9 <= recall_level <= xs[i + 1] + 1e-9:
            if abs(xs[i + 1] - xs[i]) <= 1e-12:
                best = i if ys[i] >= ys[i + 1] else i + 1
                return {
                    "qps": float(ys[best]),
                    "gamma_low": float(gammas[best]),
                    "gamma_high": float(gammas[best]),
                    "mix_high": 0.0,
                }
            fraction = (recall_level - xs[i]) / (xs[i + 1] - xs[i])
            mean_latency = (1.0 - fraction) / ys[i] + fraction / ys[i + 1]
            return {
                "qps": float(1.0 / mean_latency),
                "gamma_low": float(gammas[i]),
                "gamma_high": float(gammas[i + 1]),
                "mix_high": float(fraction),
            }
    return {
        "qps": float(ys[-1]),
        "gamma_low": float(gammas[-1]),
        "gamma_high": float(gammas[-1]),
        "mix_high": 0.0,
    }


def interpolate_frontier_qps(frontier: pd.DataFrame, recall_level: float):
    """QPS convenience wrapper for callers that do not need mixture metadata."""
    point = interpolate_frontier_point(frontier, recall_level)
    return None if point is None else point["qps"]


def matched_recall_comparison(paretos: dict[str, pd.DataFrame], recall_target: float) -> pd.DataFrame:
    """Compare all study variants only over their common average-recall support."""
    required = {"sigma_only", "estimated", "exact"}
    missing = required - set(paretos)
    if missing:
        raise ValueError(f"missing GLS contribution variants {sorted(missing)}")
    frontiers = {name: pareto_envelope_frame(paretos[name]) for name in required}
    lower = max(float(frame["avg_recall"].min()) for frame in frontiers.values())
    upper = min(float(frame["avg_recall"].max()) for frame in frontiers.values())
    if lower > upper + 1e-9:
        raise ValueError(
            f"study variants have no common average-recall support: lower={lower}, upper={upper}"
        )

    levels = np.linspace(lower, upper, 21) if upper - lower > 1e-12 else np.array([lower])
    if lower - 1e-9 <= recall_target <= upper + 1e-9:
        levels = np.append(levels, recall_target)
    levels = np.unique(np.round(levels, 12))

    rows = []
    for level in levels:
        points = {
            variant: interpolate_frontier_point(frontiers[variant], float(level))
            for variant in ("sigma_only", "estimated", "exact")
        }
        sigma_qps = points["sigma_only"]["qps"]
        estimated_qps = points["estimated"]["qps"]
        exact_qps = points["exact"]["qps"]
        row = {
            "recall_level": float(level),
            "sigma_only_qps": sigma_qps,
            "estimated_gls_qps": estimated_qps,
            "exact_gls_qps": exact_qps,
            "estimated_vs_sigma_speedup": estimated_qps / sigma_qps,
            "exact_vs_sigma_speedup": exact_qps / sigma_qps,
            "estimated_vs_exact_qps": estimated_qps / exact_qps,
        }
        for variant, prefix in (
            ("sigma_only", "sigma_only"),
            ("estimated", "estimated_gls"),
            ("exact", "exact_gls"),
        ):
            row[f"{prefix}_gamma_low"] = points[variant]["gamma_low"]
            row[f"{prefix}_gamma_high"] = points[variant]["gamma_high"]
            row[f"{prefix}_mix_high"] = points[variant]["mix_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_gls_contribution(
    paretos: dict[str, pd.DataFrame],
    out_png: Path,
    recall_target: float,
    k10_footnote: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for variant in ("sigma_only", "estimated", "exact"):
        frontier = pareto_envelope_frame(paretos[variant])
        ax.plot(
            frontier["avg_recall"],
            frontier["qps"],
            "o-",
            color=VARIANT_COLORS[variant],
            ms=4,
            label=VARIANT_DISPLAY[variant],
        )
    rt = format_recall_target(recall_target)
    ax.axvline(recall_target, color="#e2585f", ls="--", lw=1, alpha=0.7, label=f"recall target {rt}")
    ax.set_yscale("log")
    ax.set_xlabel("Average recall of chosen plans")
    ax.set_ylabel("Search-only throughput [QPS, log]")
    ax.set_title(f"GLS contribution to query optimization (recall target {rt})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.text(
        0.5,
        -0.09,
        "Offline held-out replay: each selected (plan, hyperparameter) uses its measured benchmark outcome.\n"
        "Curves show non-dominated gamma-sweep points; GLS estimation and optimizer inference time are not charged. "
        "Exact GLS is a non-deployable information reference and need not empirically dominate after model fitting.",
        fontsize=7,
        ha="center",
    )
    apply_k10_footnote(fig, k10_footnote)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved GLS contribution plot -> {out_png}")


def write_gls_contribution_outputs(
    variant_results: dict[str, dict],
    out: Path,
    recall_target: float,
    k10_footnote: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write long-form gamma points, matched-recall values, and the study figure."""
    required = {"sigma_only", "estimated", "exact"}
    missing = required - set(variant_results)
    if missing:
        raise ValueError(f"missing GLS contribution results {sorted(missing)}")

    paretos = {}
    long_frames = []
    for variant in ("sigma_only", "estimated", "exact"):
        result = variant_results[variant]
        frame = result["pareto"].copy()
        frame.insert(0, "feature_set", result["feature_set"])
        frame.insert(0, "variant", variant)
        long_frames.append(frame)
        paretos[variant] = result["pareto"]
    long_form = pd.concat(long_frames, ignore_index=True)
    long_form.to_csv(out / "gls_contribution_pareto.csv", index=False)

    matched = matched_recall_comparison(paretos, recall_target)
    matched.to_csv(out / "gls_contribution_matched_recall.csv", index=False)
    plot_gls_contribution(
        paretos,
        out / "gls_contribution_pareto.png",
        recall_target,
        k10_footnote=k10_footnote,
    )
    return long_form, matched


def plot_gls_contribution_summary(summary: pd.DataFrame, out_png: Path) -> None:
    """Cross-workload heatmaps of matched-recall QPS relative to sigma-only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    required = {
        "workload",
        "recall_target",
        "estimated_vs_sigma_speedup",
        "exact_vs_sigma_speedup",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"GLS contribution summary is missing columns {sorted(missing)}")

    preferred = [
        "flex_movies",
        "flex_reviews",
        "flex_pooled",
        "hard_movies",
        "hard_reviews",
        "hard_pooled",
        "superhard_movies",
        "superhard_reviews",
        "superhard_pooled",
    ]
    present = set(summary["workload"])
    workloads = [name for name in preferred if name in present]
    workloads.extend(sorted(present - set(workloads)))
    targets = sorted(float(v) for v in summary["recall_target"].unique())

    matrices = []
    for column in ("estimated_vs_sigma_speedup", "exact_vs_sigma_speedup"):
        matrix = (
            summary.pivot(index="workload", columns="recall_target", values=column)
            .reindex(index=workloads, columns=targets)
            .to_numpy(float)
        )
        matrices.append(matrix)
    all_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    max_deviation = max(0.05, float(np.max(np.abs(all_values - 1.0))))
    norm = TwoSlopeNorm(
        vmin=max(0.0, 1.0 - max_deviation),
        vcenter=1.0,
        vmax=1.0 + max_deviation,
    )
    approximate = np.zeros((len(workloads), len(targets)), dtype=bool)
    if "comparison_level_kind" in summary:
        approximate = (
            summary.assign(
                approximate=summary["comparison_level_kind"].ne("active_target")
            )
            .pivot(index="workload", columns="recall_target", values="approximate")
            .reindex(index=workloads, columns=targets, fill_value=False)
            .to_numpy(bool)
        )

    figure_height = max(6.2, 1.8 + 0.7 * len(workloads))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, figure_height), sharey=True)
    titles = [
        "Estimated GLS / selectivity-only",
        "Exact GLS reference / selectivity-only",
    ]
    image = None
    for ax, matrix, title in zip(axes, matrices, titles):
        image = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(targets)))
        ax.set_xticklabels([f"{target:.2f}" for target in targets])
        ax.set_yticks(np.arange(len(workloads)))
        ax.set_yticklabels([name.replace("_", " ") for name in workloads])
        ax.set_xlabel("Recall target")
        ax.set_title(title)
        for row in range(len(workloads)):
            for col in range(len(targets)):
                value = matrix[row, col]
                if np.isfinite(value):
                    marker = "†" if approximate[row, col] else ""
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}x{marker}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
    axes[0].set_ylabel("Workload")
    fig.suptitle("GLS contribution at matched average recall")
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.14, top=0.86, wspace=0.14)
    colorbar_axis = fig.add_axes([0.89, 0.20, 0.018, 0.60])
    fig.colorbar(image, cax=colorbar_axis, label="QPS ratio vs selectivity-only")
    fig.text(
        0.01,
        0.01,
        "Values above 1 favor GLS. † Target lies outside a common frontier, so the nearest common recall "
        "is used. Matched points use latency-space mixtures; planning and GLS-estimation time are excluded.",
        fontsize=8,
        ha="left",
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved aggregate GLS contribution plot -> {out_png}")


def _delta_row(quantile, delta=DEFAULT_DELTA):
    """Sweep row nearest the requested miss budget."""
    if quantile is None or len(quantile) == 0:
        return None
    idx = (quantile["delta"] - delta).abs().idxmin()
    return quantile.loc[idx]


def plot_benefits(records, ops, pareto, out_png, label, recall_target: float = RECALL_TARGET,
                  k10_footnote: bool = False, quantile=None):
    """Two panels: (A) total-latency bars vs baselines + oracle, each annotated
    with its recall-miss rate; (B) the optimizer's latency-vs-recall-safety
    frontier with the always-BF / tuned always-HNSW-pre / tuned always-IVF-pre
    / oracle reference points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    bf_qps = n / (sum(r["bf_rt"] for r in records) / 1000.0)
    oracle_qps = n / (sum(r["oracle_cost"] for r in records) / 1000.0)
    hnsw_qps = n / (sum(r["hnsw"]["rt"] for r in records) / 1000.0)
    ivf_qps = n / (sum(r["ivf"]["rt"] for r in records) / 1000.0)
    oracle_rec = float(np.mean([r["oracle_recall"] for r in records]))
    hnsw_rec = float(np.mean([r["hnsw"]["rec"] for r in records]))
    ivf_rec = float(np.mean([r["ivf"]["rec"] for r in records]))
    bf_rec = float(np.mean([r["cands"]["BF"]["rec"] for r in records]))
    d, lb = ops["default"], ops["+lambda+bump"]
    rt = format_recall_target(recall_target)
    frontier = pareto_envelope_frame(pareto)
    qrow = _delta_row(quantile)

    bars = [  # (label, qps, avg_recall, color, miss rate or None)
        ("always-BF", bf_qps, bf_rec, "#e2585f", None),
        ("oracle", oracle_qps, oracle_rec, "#39c07a", None),
        ("always\nHNSW-pre\n(tuned)", hnsw_qps, hnsw_rec, "#e6a23c", None),
        ("always\nIVF-pre\n(tuned)", ivf_qps, ivf_rec, "#2a9d8f", None),
        (f"mean loss\n(γ={d['gamma']:g})", d["qps"], d["avg_recall"], "#5aa0ff",
         d["recall_fail_rate"]),
    ]
    if qrow is not None:
        bars.append((
            f"pinball\n(δ={qrow['delta']:g})",
            float(qrow["qps"]), float(qrow["avg_recall"]), "#1d4e89",
            float(qrow["recall_fail_rate"]),
        ))
    else:
        bars.append((
            "optimizer\n(+λ+bump)", lb["qps"], lb["avg_recall"], "#b07ad6",
            lb["recall_fail_rate"],
        ))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.2, 6.2))

    # Panel A -- throughput bars (log) labelled with average recall
    x = np.arange(len(bars))
    vals = [b[1] for b in bars]
    ax1.bar(x, vals, color=[b[3] for b in bars])
    ax1.set_yscale("log")
    ax1.set_ylim(top=max(vals) * 4.2)
    ax1.set_ylabel("Search-only throughput [QPS, log]")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax1.set_title(f"Throughput, n={n}, recall target {rt}")
    for xi, b in zip(x, bars):
        label_txt = f"{b[1]:.0f} QPS\nrecall {b[2]:.3f}"
        if b[4] is not None:
            label_txt += f"\nmiss {100 * b[4]:.0f}%"
        ax1.text(xi, b[1] * 1.08, label_txt, ha="center", va="bottom", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel B -- throughput vs average-recall frontier + reference points
    ax2.plot(frontier["avg_recall"], frontier["qps"], "o-", color="#5aa0ff", ms=4,
             label="mean loss (non-dominated γ)", zorder=3)
    labelled = _label_rows(frontier, max_labels=3)
    for i, (_, r) in enumerate(frontier.iterrows()):
        if i not in labelled:
            continue
        ax2.annotate(
            f"γ={r['gamma']:g}", (r["avg_recall"], r["qps"]),
            textcoords="offset points",
            xytext=(-28, 10) if i == 0 else (6, 8),
            fontsize=7, color="#3d7ec8", ha="right" if i == 0 else "left",
        )
    if quantile is not None and len(quantile):
        qfront = pareto_envelope_frame(quantile)
        ax2.plot(qfront["avg_recall"], qfront["qps"], "s-", color="#1d4e89", ms=5,
                 label="pinball (non-dominated δ)", zorder=4)
        q_labelled = _label_rows(qfront, max_labels=3)
        for i, (_, r) in enumerate(qfront.iterrows()):
            if i not in q_labelled:
                continue
            ax2.annotate(
                f"δ={r['delta']:g}", (r["avg_recall"], r["qps"]),
                textcoords="offset points",
                xytext=(8, -16) if i == 0 else (6, -12),
                fontsize=7, color="#1d4e89",
            )
    ax2.scatter([bf_rec], [bf_qps], marker="s", s=90, color="#e2585f",
                label="always-BF", zorder=4)
    ax2.scatter([oracle_rec], [oracle_qps], marker="D", s=90, color="#39c07a",
                label="oracle", zorder=4)
    ax2.scatter([hnsw_rec], [hnsw_qps], marker="^", s=90, color="#e6a23c",
                label="always-HNSW-pre (tuned)", zorder=4)
    ax2.scatter([ivf_rec], [ivf_qps], marker="v", s=90, color="#2a9d8f",
                label="always-IVF-pre (tuned)", zorder=4)
    if qrow is None:
        ax2.scatter([lb["avg_recall"]], [lb["qps"]], marker="*", s=180,
                    color="#b07ad6", label="optimizer +λ+bump", zorder=5)
    ax2.axvline(recall_target, color="#888", ls="--", lw=1, alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_xlabel("Average recall of chosen plans")
    ax2.set_ylabel("Search-only throughput [QPS, log]")
    ax2.set_title(f"Frontier, recall target {rt}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="lower left")
    fig.suptitle(label, fontsize=13, y=1.02)

    fig.tight_layout()
    k10_line = (" " + K10_RECALL_FOOTNOTE) if k10_footnote else ""
    fig.text(
        0.01, -0.06,
        "Tuned baselines use, per query, the lowest swept hp with recall >= "
        f"{rt} (the largest swept hp if none reaches it). "
        "Oracle is the fastest plan that reaches the target; BF is always a candidate. "
        "Decision time is not charged. Mean loss predicts the average log budget and sweeps γ. "
        "Pinball predicts the (1−δ) quantile of the minimum sufficient budget and drops a plan "
        "whose quantile is infeasible. δ is a nominal miss budget, not a calibrated guarantee."
        + k10_line,
        fontsize=7.5, ha="left", va="top", wrap=True,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benefits plot -> {out_png}")


def plot_robustness(records, ops, pareto, out_png, label, recall_target: float = RECALL_TARGET,
                    k10_footnote: bool = False):
    """Robustness view: how bad do the recall misses get?

    Panel A: CDF of realised recall for the optimizer operating points and the
    tuned always-HNSW-pre / tuned always-IVF-pre / always-BF / oracle
    baselines. A curve hugging the bottom-
    right corner is robust; mass to the left of a threshold line is the share
    of queries with recall below it.
    Panel B: share of pairs below each recall threshold across the gamma sweep.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = [
        ("optimizer (default)", ops["default"]["recall"], "#5aa0ff"),
        ("optimizer (+lam+bump)", ops["+lambda+bump"]["recall"], "#b07ad6"),
        ("always-HNSW-pre (tuned)", np.array([r["hnsw"]["rec"] for r in records]), "#e6a23c"),
        ("always-IVF-pre (tuned)", np.array([r["ivf"]["rec"] for r in records]), "#2a9d8f"),
        ("always-BF", np.array([r["cands"]["BF"]["rec"] for r in records]), "#e2585f"),
        ("oracle", np.array([r["oracle_recall"] for r in records]), "#39c07a"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    thresholds = robustness_thresholds(recall_target)

    for name, rec, color in curves:
        xs = np.sort(rec)
        cdf = np.arange(1, len(xs) + 1) / len(xs)
        ax1.step(xs, cdf, where="post", color=color, lw=2,
                 label=f"{name} (mean {rec.mean():.3f})")
    for thr in thresholds:
        ax1.axvline(thr, color="#888", ls="--", lw=0.8, alpha=0.6)
        ax1.text(thr, 1.02, f"{thr:g}", ha="center", fontsize=8, color="#666")
    ax1.set_xlabel("Realised recall of chosen plan")
    ax1.set_ylabel("CDF (share of pairs with recall <= x)")
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left")
    rt = format_recall_target(recall_target)
    ax1.set_title(f"Realised-recall CDF over {len(records)} pairs -- {label} (recall ≥ {rt})")

    shades = {
        0.99: "#c6dbef",
        0.95: "#9ecae1",
        0.90: "#6baed6",
        0.80: "#3182bd",
        0.50: "#08519c",
    }
    for thr in thresholds:
        col = _bad_rate_key(thr, recall_target)
        if col not in pareto.columns:
            continue
        ax2.plot(pareto["gamma"], 100 * pareto[col], "o-",
                 color=shades.get(round(thr, 2), "#3182bd"), label=f"recall < {thr:g}")
    ax2.set_xscale("log")
    ax2.set_xlabel("gamma (index margin, log)")
    ax2.set_ylabel("Share of pairs with bad recall [%]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title(f"Bad-recall share vs gamma -- {label}")

    fig.tight_layout()
    apply_k10_footnote(fig, k10_footnote)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved robustness plot -> {out_png}")


def plot_decision_scatter(records, chosen, feat, out_png, label, gamma,
                          recall_target: float = RECALL_TARGET, k10_footnote: bool = False):
    """(selectivity, rho) scatter coloured by the LOGICAL plan chosen, split into
    optimizer (top) vs ground-truth oracle (bottom) x movies vs reviews. Mirrors
    the best_plan_scatter plots in analysis/plots/query_optimizer/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.DataFrame([
        {"sigma": feat[r["pair"]][0], "rho": feat[r["pair"]][1],
         "query_type": r["pair"][0], "optimizer": ch, "oracle": r["oracle_plan"]}
        for r, ch in zip(records, chosen)])

    datasets = [qt for qt in ["movies", "reviews", "cases"] if (d["query_type"] == qt).any()]
    fig, axes = plt.subplots(2, len(datasets), figsize=(7.5 * len(datasets), 11),
                             sharex=True, sharey=True, squeeze=False)
    for j, qt in enumerate(datasets):
        sub = d[d["query_type"] == qt]
        for i, key in enumerate(["optimizer", "oracle"]):
            ax = axes[i][j]
            for plan in ALL_PLANS:
                p = sub[sub[key] == plan]
                if len(p) == 0:
                    continue
                ax.scatter(p["sigma"], p["rho"], s=22, alpha=0.6,
                           c=PLAN_PLOT_COLORS[plan], edgecolors="none",
                           label=f"{plan} ({len(p)})")
            ax.set_xscale("log")
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
            ax.grid(True, alpha=0.3)
            n_ds = len(sub)
            tag = f"optimizer (g={gamma:g})" if key == "optimizer" else "oracle (ground truth)"
            ax.set_title(f"{tag} -- {qt} (n={n_ds})")
            if i == 1:
                ax.set_xlabel("Filter selectivity (log)")
            if j == 0:
                ax.set_ylabel("GLS correlation rho")
            ax.legend(fontsize=7, loc="lower left", title="plan (count)", framealpha=0.9)
    fig.suptitle(
        f"Logical-plan decisions across (selectivity, rho) -- {label} "
        f"(recall ≥ {format_recall_target(recall_target)})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    apply_k10_footnote(fig, k10_footnote)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved decision scatter -> {out_png}")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _run_one_core(
    path,
    variant,
    out: Path,
    gamma,
    lam,
    min_leaf,
    recall_target: float = RECALL_TARGET,
    feature_set: str = FEATURE_SIGMA_RHO,
    sweep: pd.DataFrame | None = None,
    query_type: str | None = None,
    grids: dict | None = None,
    dataset_grids: dict | None = None,
    train_test_splits=None,
    eval_pairs_override: set | None = None,
):
    out.mkdir(parents=True, exist_ok=True)
    _check_feature_set(feature_set)
    display_label = VARIANT_DISPLAY.get(variant, variant)
    print(f"\n{'='*74}\n VARIANT = {display_label}   ({path}) -> {out}\n{'='*74}")
    rt = format_recall_target(recall_target)
    sweep = load_sweep(path, query_type=query_type) if sweep is None else sweep.copy()
    k10 = df_is_k10_only(sweep)
    if grids is None:
        grids = derive_grids(sweep)
    if dataset_grids is None:
        dataset_grids = derive_dataset_grids(sweep)
    print("  per-(dataset,k)/plan hp grids (derived from swept CSV):")
    for plan in ANN_PLANS:
        by_dataset = {
            f"{query_type}/k={int(k)}": dataset_grids[(query_type, int(k), plan)]
            for query_type, k in (
                sweep[["query_type", "k"]]
                .drop_duplicates()
                .sort_values(["query_type", "k"])
                .itertuples(index=False, name=None)
            )
        }
        print(f"    {plan}: {by_dataset}")
    pp = build_pairplan(sweep, recall_target)
    meas = build_measure(sweep)
    all_pairs = [tuple(x) for x in pp[PAIR_KEYS].drop_duplicates().to_numpy()]
    all_pairs = [(p[0], int(p[1]), int(p[2]), int(p[3])) for p in all_pairs]
    pairs = (
        [p for p in all_pairs if p in eval_pairs_override]
        if eval_pairs_override is not None
        else all_pairs
    )
    oracle = build_oracle(meas, all_pairs, recall_target)
    print(f"  {len(pairs)} evaluated (query,filter) pairs; plans: {sorted(pp['plan'].unique())}")
    print(f"  per-plan feasibility base rate (fraction reaching recall>={rt}):")
    base = pp.groupby("plan")["feasible"].mean().reindex(ALL_PLANS)
    print("    " + "  ".join(f"{p}={v:.3f}" for p, v in base.items()))
    for plan in ("HNSW-pre", "IVF-pre"):
        tuned_hps = [tuned_baseline(meas, pair, plan, recall_target)["hp"]
                     for pair in pairs if (*pair, plan) in meas]
        if tuned_hps:
            vc = pd.Series(tuned_hps).value_counts().sort_index()
            print(f"  tuned always-{plan} baseline hp distribution: "
                  + ", ".join(f"{int(h)}x{int(c)}" for h, c in vc.items()))

    feat_rows = pp.drop_duplicates(PAIR_KEYS, keep="first")
    feat = {
        (r.query_type, int(r.query_id_num), int(r.filter_id), int(r.k)): (
            float(r.sigma), float(r.rho), r.query_type
        )
        for r in feat_rows.itertuples(index=False)
    }

    records, diag = run_cv(
        sweep,
        pp,
        meas,
        oracle,
        grids,
        min_leaf=min_leaf,
        recall_target=recall_target,
        feature_set=feature_set,
        train_test_splits=train_test_splits,
        dataset_grids=dataset_grids,
    )
    quantile = run_quantile_policy(
        sweep,
        pp,
        meas,
        oracle,
        grids,
        min_leaf,
        recall_target,
        feature_set,
        train_test_splits=train_test_splits,
        dataset_grids=dataset_grids,
    )
    quantile.to_csv(out / f"quantile_pareto_{variant}.csv", index=False)
    print("\n--- Pinball budget (quantile of the minimum sufficient hyperparameter) ---")
    print(quantile[[
        "delta", "avg_recall", "recall_fail_rate", "qps", "bf_share",
        "speedup_vs_bf", "chosen_is_oracle_pct",
    ]].to_string(index=False, formatters={
        "avg_recall": lambda x: f"{x:.3f}",
        "recall_fail_rate": lambda x: f"{100 * x:.1f}%",
        "qps": lambda x: f"{x:.1f}",
        "bf_share": lambda x: f"{100 * x:.0f}%",
        "speedup_vs_bf": lambda x: f"{x:.2f}x",
        "chosen_is_oracle_pct": lambda x: f"{x:.0f}%",
    }))

    pd.DataFrame(report_stages(diag)).to_csv(out / f"stage12_metrics_{variant}.csv", index=False)
    pd.DataFrame(report_s3(diag)).to_csv(out / f"stage3_metrics_{variant}.csv", index=False)

    extra_rate = f"bad_recall_rate_{round(float(recall_target) * 100):03d}"
    knob_keys = [
        "gamma", "lam", "bump", "avg_recall", "qps", "recall_fail_rate", "recall_p10",
        "bad_recall_rate_090", "bad_recall_rate_080", "bad_recall_rate_050",
        "chosen_is_oracle_pct",
        "mean_regret", "total_realised_ms", "speedup_vs_bf",
        "speedup_vs_hnsw", "speedup_vs_ivf",
        "hnsw_qps", "hnsw_avg_recall", "ivf_qps", "ivf_avg_recall",
    ]
    if extra_rate not in knob_keys:
        knob_keys.append(extra_rate)

    # operating points: default / +lambda / +safety-bump / +both
    print("\n--- Operating points (gamma={:g}) ---".format(gamma))
    ops = {
        "default": optimizer_metrics(records, gamma, 0.0, False, recall_target),
        "+lambda": optimizer_metrics(records, gamma, lam, False, recall_target),
        "+safety_bump": optimizer_metrics(records, gamma, 0.0, True, recall_target),
        "+lambda+bump": optimizer_metrics(records, gamma, lam, True, recall_target),
    }
    for name, m in ops.items():
        report_optimizer(m, f"{display_label} / {name}", recall_target)
    knob_rows = [{"variant": variant, "feature_set": feature_set, "operating_point": name,
                  **{k: m[k] for k in knob_keys if k in m}}
                 for name, m in ops.items()]
    pd.DataFrame(knob_rows).to_csv(out / f"operating_points_{variant}.csv", index=False)

    # per-pair selection dump at the default operating point
    m = ops["default"]
    sel = build_selection_frame(
        records, m, feat, ops["+safety_bump"], variant, feature_set
    )
    sel.to_csv(out / f"selection_{variant}.csv", index=False)

    # per-dataset breakdown at the default operating point
    sel["is_oracle"] = [c == r["oracle_plan"] for c, r in zip(m["chosen"], records)]
    print(f"\n--- Per-dataset breakdown [{display_label}, default gamma={gamma:g}] ---")
    print(f"  {'dataset':8s} {'n':>5s} {'QPS':>7s} {'BF_QPS':>7s} "
          f"{'speedup':>7s} {'avg_recall':>10s} {'miss':>6s} {'==oracle':>8s}")
    for ds, g in sel.groupby("query_type"):
        qps = len(g) / (g["realised_ms"].sum() / 1000)
        bf_qps = len(g) / (g["bf_ms"].sum() / 1000)
        print(f"  {ds:8s} {len(g):5d} {qps:7.1f} {bf_qps:7.1f} {qps/bf_qps:6.1f}x "
              f"{g['recall'].mean():10.4f} {100*g['recall_fail'].mean():5.1f}% "
              f"{100*g['is_oracle'].mean():7.1f}%")
    plot_decision_scatter(
        records, m["chosen"], feat,
        out / f"decision_scatter_{variant}.png", display_label, gamma,
        recall_target=recall_target, k10_footnote=k10,
    )

    pareto_keys = [
        "gamma", "lam", "bump", "avg_recall", "qps", "recall_fail_rate",
        "recall_p10", "bad_recall_rate_090", "bad_recall_rate_080",
        "bad_recall_rate_050",
        "chosen_is_oracle_pct", "mean_regret", "median_regret", "total_realised_ms",
        "total_oracle_ms", "total_bf_ms", "total_hnsw_ms", "total_ivf_ms",
        "hnsw_qps", "hnsw_avg_recall", "ivf_qps", "ivf_avg_recall",
        "speedup_vs_bf", "speedup_vs_hnsw", "speedup_vs_ivf",
    ]
    if extra_rate not in pareto_keys:
        pareto_keys.append(extra_rate)

    # Gamma operating points (lambda=0, no bump). Plotting and matched-recall
    # comparisons retain only the non-dominated envelope.
    pareto = pd.DataFrame([{k: mm[k] for k in pareto_keys if k in mm}
        for mm in (optimizer_metrics(records, g, 0.0, False, recall_target) for g in GAMMA_SWEEP)])
    pareto.to_csv(out / f"gamma_pareto_{variant}.csv", index=False)
    print(f"\n--- gamma-sweep operating points [{display_label}] "
          "(search-only throughput vs average recall) ---")
    print(pareto[["gamma", "avg_recall", "qps", "recall_fail_rate",
                  "speedup_vs_bf", "mean_regret"]].to_string(
        index=False, formatters={
            "avg_recall": lambda x: f"{x:.4f}",
            "qps": lambda x: f"{x:.1f}",
            "recall_fail_rate": lambda x: f"{100*x:.1f}%",
            "speedup_vs_bf": lambda x: f"{x:.2f}x",
            "mean_regret": lambda x: f"{x:.3f}"}))
    plot_pareto(
        pareto, out / f"gamma_pareto_{variant}.png", display_label,
        recall_target=recall_target, k10_footnote=k10,
    )
    safety_frontier = safety_pareto_envelope_frame(pareto)
    safety_frontier.to_csv(out / f"gamma_safety_pareto_{variant}.csv", index=False)
    plot_safety_pareto(
        pareto,
        out / f"gamma_safety_pareto_{variant}.png",
        display_label,
        recall_target=recall_target,
        k10_footnote=k10,
    )
    plot_benefits(
        records, ops, pareto, out / f"benefits_{variant}.png", display_label,
        recall_target=recall_target, k10_footnote=k10, quantile=quantile,
    )
    plot_robustness(
        records, ops, pareto, out / f"robustness_{variant}.png", display_label,
        recall_target=recall_target, k10_footnote=k10,
    )

    pairs_ordered = [r["pair"] for r in records]
    curves = pd.concat(
        [static_plan_curve(meas, pairs_ordered, plan) for plan in PHYSICAL_PLANS],
        ignore_index=True,
    )
    physical = {
        plan: physical_optimizer_metrics(records, plan, recall_target)
        for plan in PHYSICAL_PLANS
    }
    print("\n--- physical-only optimizer (logical plan fixed; predicted hp only) ---")
    for plan, pm in physical.items():
        print(f"  {plan:10s} ({pm['hp_name']:9s})  {pm['qps']:7.1f} QPS  "
              f"avg recall {pm['avg_recall']:.4f}  "
              f"recall<{format_recall_target(recall_target)} on {100 * pm['recall_fail_rate']:.1f}%")
    point_rows = [
        {
            "kind": "point",
            "name": "always-BF",
            "plan": "BF",
            "hyperparam": 0.0,
            "hp_name": "",
            "avg_recall": float(np.mean([r["cands"]["BF"]["rec"] for r in records])),
            "qps": _aggregate_qps([r["bf_rt"] for r in records]),
            "n": len(records),
        },
        {
            "kind": "point",
            "name": "oracle",
            "plan": "",
            "hyperparam": np.nan,
            "hp_name": "",
            "avg_recall": float(np.mean([r["oracle_recall"] for r in records])),
            "qps": _aggregate_qps([r["oracle_cost"] for r in records]),
            "n": len(records),
        },
        {
            "kind": "point",
            "name": "optimizer",
            "plan": "",
            "hyperparam": np.nan,
            "hp_name": "",
            "avg_recall": ops["default"]["avg_recall"],
            "qps": ops["default"]["qps"],
            "n": len(records),
        },
    ]
    for plan, pm in physical.items():
        point_rows.append({
            "kind": "point",
            "name": f"physical-{plan}",
            "plan": plan,
            "hyperparam": np.nan,
            "hp_name": pm["hp_name"],
            "avg_recall": pm["avg_recall"],
            "qps": pm["qps"],
            "n": pm["n"],
        })
    plan_table = pd.concat([curves, pd.DataFrame(point_rows)], ignore_index=True)
    plan_table.to_csv(out / f"qps_recall_plans_{variant}.csv", index=False)
    scatter_points = _qps_recall_scatter_points(
        point_rows, ops["default"]["gamma"]
    )
    plot_qps_recall_plans(
        curves,
        scatter_points,
        out / f"qps_recall_plans_{variant}.png",
        display_label,
        recall_target=recall_target,
        k10_footnote=k10,
    )
    return {
        "default": ops["default"],
        "ops": ops,
        "pareto": pareto,
        "selection": sel,
        "records": records,
        "feature_set": feature_set,
        "k10": k10,
    }


def run_one(
    path,
    variant,
    out,
    gamma,
    lam,
    min_leaf,
    recall_target: float = RECALL_TARGET,
    feature_set: str = FEATURE_SIGMA_RHO,
    sweep: pd.DataFrame | None = None,
    query_type: str | None = None,
):
    sweep_df = load_sweep(path, query_type=query_type) if sweep is None else sweep.copy()
    has_split = (
        "split" in sweep_df.columns
        and set(sweep_df["split"].dropna().astype(str).unique()) == {"train", "test"}
    )
    if not has_split:
        return _run_one_core(
            path,
            variant,
            Path(out),
            gamma,
            lam,
            min_leaf,
            recall_target=recall_target,
            feature_set=feature_set,
            sweep=sweep_df,
            query_type=query_type,
        )

    out_path = Path(out)
    train_sweep = sweep_df[sweep_df["split"] == "train"].copy()
    test_sweep = sweep_df[sweep_df["split"] == "test"].copy()
    tr_ids = set(train_sweep["query_id_num"].astype(int))
    te_ids = set(test_sweep["query_id_num"].astype(int))
    if not tr_ids or not te_ids:
        raise ValueError(f"{path}: split column must have non-empty train and test sets")
    if tr_ids & te_ids:
        raise ValueError(f"{path}: train and test query_id_num sets overlap")

    grids = derive_grids(sweep_df)
    dataset_grids = derive_dataset_grids(sweep_df)

    # 1. Grouped 5-fold CV on training queries only -> rXXX/train_cv/
    train_cv_out = out_path / "train_cv"
    train_cv_res = _run_one_core(
        path,
        variant,
        train_cv_out,
        gamma,
        lam,
        min_leaf,
        recall_target=recall_target,
        feature_set=feature_set,
        sweep=train_sweep,
        query_type=query_type,
        grids=grids,
        dataset_grids=dataset_grids,
        train_test_splits=None,
    )

    # 2. One-shot refit on all training queries, scored once on all test queries -> rXXX/
    pp_train = build_pairplan(train_sweep, recall_target)
    pp_test = build_pairplan(test_sweep, recall_target)
    test_pairs = {
        (p[0], int(p[1]), int(p[2]), int(p[3]))
        for p in pp_test[PAIR_KEYS].drop_duplicates().to_numpy()
    }
    test_res = _run_one_core(
        path,
        variant,
        out_path,
        gamma,
        lam,
        min_leaf,
        recall_target=recall_target,
        feature_set=feature_set,
        sweep=sweep_df,
        query_type=query_type,
        grids=grids,
        dataset_grids=dataset_grids,
        train_test_splits=[(pp_train, pp_test, train_sweep, test_sweep)],
        eval_pairs_override=test_pairs,
    )
    test_res["train_cv"] = train_cv_res
    return test_res


def _write_single_contribution_summary(
    variant_results: dict[str, dict],
    out_dir: Path,
    recall_target: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, matched = write_gls_contribution_outputs(
        variant_results,
        out_dir,
        recall_target,
        k10_footnote=variant_results["exact"]["k10"],
    )
    at_target = matched[np.isclose(matched["recall_level"], recall_target)]
    if len(at_target):
        comparison = at_target.iloc[0].copy()
        comparison_kind = "active_target"
    else:
        nearest = (matched["recall_level"] - recall_target).abs().idxmin()
        comparison = matched.loc[nearest].copy()
        comparison_kind = "nearest_common_recall"
    comparison["recall_target"] = recall_target
    comparison["comparison_level_kind"] = comparison_kind
    pd.DataFrame([comparison]).to_csv(out_dir / "gls_contribution_summary.csv", index=False)
    return matched, comparison


def _maybe_write_caselaw_rollup(out_root: Path) -> None:
    """When CaseLaw recall-target directories exist under out_root, write the CaseLaw rollup."""
    all_matched = []
    summary_rows = []
    for target in (0.80, 0.90, 0.95, 0.99):
        tag = recall_tag(target)
        tag_dir = out_root / tag
        m_path = tag_dir / "gls_contribution_matched_recall.csv"
        s_path = tag_dir / "gls_contribution_summary.csv"
        if not m_path.is_file() or not s_path.is_file():
            continue
        m_df = pd.read_csv(m_path)
        m_df.insert(0, "recall_target", target)
        m_df.insert(0, "workload", "native_cases")
        all_matched.append(m_df)

        s_df = pd.read_csv(s_path)
        s_df.insert(0, "workload", "native_cases")
        summary_rows.append(s_df)

    if not summary_rows:
        return
    matched_cat = pd.concat(all_matched, ignore_index=True)
    summary_cat = pd.concat(summary_rows, ignore_index=True)
    matched_cat.to_csv(out_root / "gls_contribution_matched_recall_all.csv", index=False)
    summary_cat.to_csv(out_root / "gls_contribution_summary.csv", index=False)
    plot_gls_contribution_summary(summary_cat, out_root / "gls_contribution_summary.png")


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results",
                    default=str(root / "plots/query_optimizer/all_query_results.csv"))
    ap.add_argument("--gls", choices=["exact", "estimated", "both"], default="both")
    ap.add_argument("--gamma", type=float, default=1.3)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--min-leaf", type=int, default=30)
    ap.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    ap.add_argument(
        "--query-type",
        choices=["all", "movies", "reviews", "cases"],
        default="all",
        help="Optionally evaluate one table from a pooled sweep CSV",
    )
    ap.add_argument("--out", default=str(root / "plots/qo_prototype"))
    ap.add_argument(
        "--gls-contribution-study",
        action="store_true",
        help=(
            "Run sigma-only, estimated-GLS, and exact-GLS variants on validated "
            "identical measurements and write a combined matched-recall comparison"
        ),
    )
    args = ap.parse_args()

    exact_path = Path(args.results)
    est_path = exact_path.parent / "gls_est" / "all_query_results.csv"
    out_root = Path(args.out)
    out = out_root / recall_tag(args.recall_target)
    out.mkdir(parents=True, exist_ok=True)
    query_type = None if args.query_type == "all" else args.query_type

    targets = []
    if args.gls_contribution_study:
        if args.gls != "both":
            ap.error("--gls-contribution-study requires --gls both")
        missing_inputs = [path for path in (exact_path, est_path) if not path.is_file()]
        if missing_inputs:
            raise SystemExit(
                "GLS contribution study requires cached exact and estimated sweeps; "
                f"missing: {[str(path) for path in missing_inputs]}"
            )
        exact_sweep = load_sweep(
            exact_path, require_rho=True, strict=True, query_type=query_type
        )
        estimated_sweep = load_sweep(
            est_path, require_rho=True, strict=True, query_type=query_type
        )
        validate_study_pair(exact_sweep, estimated_sweep)
        est_features = estimated_feature_set(estimated_sweep.columns)
        targets = [
            ("sigma_only", exact_path, exact_sweep, FEATURE_SIGMA_ONLY),
            ("estimated", est_path, estimated_sweep, est_features),
            ("exact", exact_path, exact_sweep, FEATURE_SIGMA_RHO),
        ]
    else:
        if args.gls in ("exact", "both"):
            if not exact_path.is_file():
                raise SystemExit(f"exact GLS CSV not found: {exact_path}")
            targets.append(("exact", exact_path, None, FEATURE_SIGMA_RHO))
        if args.gls in ("estimated", "both"):
            if est_path.is_file():
                est_features = estimated_feature_set(pd.read_csv(est_path, nrows=0).columns)
                targets.append(("estimated", est_path, None, est_features))
            elif args.gls == "estimated":
                raise SystemExit(f"estimated GLS CSV not found: {est_path}")
            else:
                print(f"(skipping estimated GLS: {est_path} not found)")

    summary = {}
    for variant, path, sweep, feature_set in targets:
        summary[variant] = run_one(
            path,
            variant,
            out,
            args.gamma,
            args.lam,
            args.min_leaf,
            args.recall_target,
            feature_set=feature_set,
            sweep=sweep,
            query_type=query_type,
        )

    if "exact" in summary and "estimated" in summary:
        e, s = summary["exact"]["default"], summary["estimated"]["default"]
        print(f"\n{'='*74}\n exact vs estimated GLS  (confirm estimated ~= exact)\n{'='*74}")
        print(f"  average recall      : exact {e['avg_recall']:.4f}   "
              f"estimated {s['avg_recall']:.4f}")
        print(f"  throughput [QPS]    : exact {e['qps']:.1f}   estimated {s['qps']:.1f}")
        print(f"  speedup vs BF       : exact {e['speedup_vs_bf']:.2f}x   "
              f"estimated {s['speedup_vs_bf']:.2f}x")
        print(f"  chosen==oracle      : exact {e['chosen_is_oracle_pct']:.1f}%   "
              f"estimated {s['chosen_is_oracle_pct']:.1f}%")

    if args.gls_contribution_study:
        if all("train_cv" in summary[v] for v in ("sigma_only", "estimated", "exact")):
            train_cv_summary = {v: summary[v]["train_cv"] for v in ("sigma_only", "estimated", "exact")}
            _write_single_contribution_summary(
                train_cv_summary, out / "train_cv", args.recall_target
            )
        _, comparison = _write_single_contribution_summary(
            summary, out, args.recall_target
        )
        print(f"\n{'='*74}\n GLS contribution at matched recall\n{'='*74}")
        print(f"  comparison level    : {comparison['recall_level']:.4f} ({comparison['comparison_level_kind']})")
        print(f"  sigma-only QPS      : {comparison['sigma_only_qps']:.1f}")
        print(f"  estimated GLS QPS   : {comparison['estimated_gls_qps']:.1f} "
              f"({comparison['estimated_vs_sigma_speedup']:.3f}x vs sigma-only)")
        print(f"  exact GLS QPS       : {comparison['exact_gls_qps']:.1f} "
              f"({comparison['exact_vs_sigma_speedup']:.3f}x vs sigma-only)")
        if query_type == "cases" or set(summary["exact"]["selection"]["query_type"].unique()) == {"cases"}:
            _maybe_write_caselaw_rollup(out_root)

    print(f"\nOutputs -> {out}")


if __name__ == "__main__":
    main()
