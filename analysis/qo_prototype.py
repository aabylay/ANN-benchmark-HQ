#!/usr/bin/env python3
"""FANNS query optimizer -- v2 prototype (FAISS only).

Design: prompts/qo_prototype.txt.

Decide, per (query vector, filter), which FAISS plan to run AND with which search
hyperparameter, using only cheap plan-time inputs (sigma, rho, N) -- no vector
search. Unlike v1 (which ramped to the tuned hp online and charged the ramp), v2
PREDICTS the hyperparameter directly, so the cost we compare is the single tuned
query. That introduces a real recall-miss risk (an under-predicted hp can miss
recall at query time), which the decision rule and the evaluation handle
explicitly.

Pipeline (three per-plan models + one decision rule):

  STAGE 1  feasibility (per ANN plan, classifier). CART (depth <= 3) on
      X = {sigma, rho, k} (pooled across tables) -> P_feasible in [0,1]. k enters
      because feasibility is per-query-k: a larger k needs a larger hp and shrinks
      the feasible region within the swept grid.

  STAGE 2  hyperparameter (per ANN plan). "Best hp" = the LOWEST swept hp that
      reaches recall >= 0.95, evaluated PER k. Two variants are fit on
      X = {sigma, rho, k}:
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
      (recall = 1.0), lambda penalises low-confidence ANN picks. Sweep gamma for
      the latency-vs-recall-safety Pareto.

Evaluation is grouped 5-fold CV (folds split *queries*, so a query's filters
never straddle train/test). The realised RECALL-FAILURE RATE (fraction of chosen
pairs whose measured recall < 0.95 at the predicted hp) is the new, critical v2
metric. Runs on both exact and estimated GLS. Outputs -> plots/qo_prototype/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RECALL_TARGET = 0.95
N_TABLE = {"movies": 551_155, "reviews": 2_598_267}

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

HNSW_GRID = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
             1200, 1400, 1600, 1800, 2000, 2250, 2500]
# IVF probes: the swept large-dataset grid, plus 500 = the reviews-only x2
# doubling of the 250 max applied in runner.py.
IVF_GRID = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            120, 140, 160, 180, 200, 225, 250, 500]
GRID = {"HNSW-pre": HNSW_GRID, "HNSW-post": HNSW_GRID,
        "IVF-pre": IVF_GRID, "IVF-post": IVF_GRID}
# A (query, filter, k) triple is the unit of decision: k is part of the query
# request (top-k), so it identifies a pair alongside the query vector and filter.
PAIR_KEYS = ["query_type", "query_id_num", "filter_id", "k"]

# wide enough to expose the Pareto: in v2 ANN is far cheaper than BF, so small
# gamma never displaces BF -- only large gamma trades latency for recall safety.
GAMMA_SWEEP = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]


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
    """Minimal CART. task='reg' (variance/mean leaf) or 'clf' (gini/p leaf)."""

    def __init__(self, task="reg", max_depth=3, min_leaf=30):
        self.task = task
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.root = None

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y, float), 0)
        return self

    def _leaf(self, y):
        node = _Node()
        node.val = float(y.mean()) if len(y) else 0.0
        return node

    def _best_split(self, X, y):
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
def load_sweep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["algorithm"].isin(PLANS)].copy()
    df["plan"] = df["algorithm"].map(PLANS)
    df["N"] = df["query_type"].map(N_TABLE).astype(float)
    df["sigma"] = df["filter_selectivity"].astype(float)
    rho = pd.to_numeric(df["gls_correlation"], errors="coerce").fillna(0.0)
    df["rho"] = np.clip(rho.to_numpy(), -0.999, 0.999)
    df["n_pass"] = np.clip(df["sigma"].to_numpy() * df["N"].to_numpy(), 1.0, None)
    df["runtime_ms"] = df["runtime"].astype(float) * 1000.0
    return df.dropna(subset=["sigma"])


def build_pairplan(sweep: pd.DataFrame) -> pd.DataFrame:
    """One row per (query, filter, plan): feasibility + best (lowest feasible) hp."""
    rows = []
    for (qt, qid, fid, kk, plan), g in sweep.groupby(PAIR_KEYS + ["plan"], sort=False):
        g = g.sort_values("hyperparam")
        recall = g["recall"].to_numpy()
        hp = g["hyperparam"].to_numpy()
        feasible = bool((recall >= RECALL_TARGET).any())
        best_hp = 0.0
        if plan != "BF" and feasible:
            best_hp = float(hp[np.argmax(recall >= RECALL_TARGET)])
        r0 = g.iloc[0]
        rows.append({
            "query_type": qt, "query_id_num": int(qid), "filter_id": int(fid),
            "k": int(kk), "plan": plan, "sigma": float(r0["sigma"]), "rho": float(r0["rho"]),
            "N": float(r0["N"]), "n_pass": float(r0["n_pass"]),
            "feasible": int(feasible), "best_hp": best_hp,
        })
    return pd.DataFrame(rows)


def build_measure(sweep: pd.DataFrame) -> dict:
    """(pair, plan) -> (hps, recall, runtime_ms) sorted ascending (ground truth)."""
    meas = {}
    for (qt, qid, fid, kk, plan), g in sweep.groupby(PAIR_KEYS + ["plan"], sort=False):
        g = g.sort_values("hyperparam")
        meas[(qt, int(qid), int(fid), int(kk), plan)] = (
            g["hyperparam"].to_numpy(float),
            g["recall"].to_numpy(float),
            g["runtime_ms"].to_numpy(float),
        )
    return meas


def measure_at(meas, key, target_hp):
    """Measured (recall, runtime_ms) at smallest available hp >= target (else max)."""
    hps, rec, rt = meas[key]
    idx = int(np.searchsorted(hps, target_hp, side="left"))
    if idx >= len(hps):
        idx = len(hps) - 1
    return float(rec[idx]), float(rt[idx])


def min_feasible(meas, key):
    """(runtime_ms, recall) at the lowest available hp reaching recall>=target, else None."""
    hps, rec, rt = meas[key]
    ok = np.where(rec >= RECALL_TARGET)[0]
    if len(ok) == 0:
        return None
    i = ok[0]
    return float(rt[i]), float(rec[i])


def build_oracle(meas, pairs) -> dict:
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
                mf = min_feasible(meas, key)
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
def stage3_feats(n_pass, rho, N, hp=None, k=None):
    cols = [np.log(n_pass), np.asarray(rho, float), np.log(N)]
    if hp is not None:
        cols.append(np.log(np.clip(np.asarray(hp, float), 1.0, None)))
    if k is not None:
        cols.append(np.log(np.clip(np.asarray(k, float), 1.0, None)))
    return np.column_stack(cols)


# Stage 1/2 CART feature columns (must match between fit and predict).
STAGE12_FEATS = ["sigma", "rho", "k"]


def fit_models(tr_pp, tr_sweep, min_leaf):
    feas_clf, hp_reg, hp_uni, s3 = {}, {}, {}, {}
    for plan in ANN_PLANS:
        tr = tr_pp[tr_pp["plan"] == plan]
        X = tr[STAGE12_FEATS].to_numpy()
        y_feas = tr["feasible"].to_numpy()
        feas_clf[plan] = CART("clf", min_leaf=min_leaf).fit(X, y_feas)
        hp_uni[plan] = CART("reg", min_leaf=min_leaf).fit(X, tr["best_hp"].to_numpy())
        fr = tr[tr["feasible"] == 1]
        if len(fr) >= 2 * min_leaf:
            hp_reg[plan] = CART("reg", min_leaf=min_leaf).fit(
                fr[STAGE12_FEATS].to_numpy(), np.log(fr["best_hp"].to_numpy()))
        else:  # too few feasible rows -> conservative constant (max grid)
            hp_reg[plan] = float(np.log(GRID[plan][-1]))
    for plan in ALL_PLANS:
        s = tr_sweep[tr_sweep["plan"] == plan]
        if len(s) == 0:
            continue
        hp = None if plan == "BF" else s["hyperparam"].to_numpy()
        s3[plan] = LogOLS().fit(
            stage3_feats(s["n_pass"].to_numpy(), s["rho"].to_numpy(), s["N"].to_numpy(),
                         hp, s["k"].to_numpy()),
            s["runtime_ms"].to_numpy())
    return feas_clf, hp_reg, hp_uni, s3


def predict_hp(model, X, grid):
    """Ablation hp: exp(log-hp regressor) snapped to grid (>0 always)."""
    if isinstance(model, float):
        raw = np.full(len(X), np.exp(model))
    else:
        raw = np.exp(model.predict(X))
    return np.array([max(snap_ceil(v, grid), grid[0]) for v in raw])


# --------------------------------------------------------------------------- #
# cross-validated evaluation
# --------------------------------------------------------------------------- #
def group_folds(pp, n_folds=5, seed=0):
    groups = pp[["query_type", "query_id_num"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = groups.iloc[rng.permutation(len(groups))].reset_index(drop=True)
    groups["fold"] = np.arange(len(groups)) % n_folds
    return pp.merge(groups, on=["query_type", "query_id_num"], how="left")


def run_cv(sweep, pp, meas, oracle, n_folds=5, seed=0, min_leaf=30):
    pp = group_folds(pp, n_folds, seed)
    fold_of = pp[PAIR_KEYS + ["fold"]].drop_duplicates()
    sweep = sweep.merge(fold_of, on=PAIR_KEYS, how="left")

    stage1 = {p: {"acc": [], "auc": []} for p in ANN_PLANS}
    stage2 = {p: {"hit": 0, "n": 0, "step": []} for p in ANN_PLANS}
    stage2u = {p: {"acc": [], "hit": 0, "n": 0} for p in ANN_PLANS}  # unified variant
    s3_true = {p: {"yt": [], "yp": []} for p in ALL_PLANS}
    s3_pred = {p: {"yt": [], "yp": []} for p in ANN_PLANS}
    records = []

    for fold in range(n_folds):
        tr_pp, te_pp = pp[pp["fold"] != fold], pp[pp["fold"] == fold]
        tr_sweep, te_sweep = sweep[sweep["fold"] != fold], sweep[sweep["fold"] == fold]
        feas_clf, hp_reg, hp_uni, s3 = fit_models(tr_pp, tr_sweep, min_leaf)

        # Stage 3 "true hp": predict every test sweep row
        for plan in ALL_PLANS:
            sub = te_sweep[te_sweep["plan"] == plan]
            if len(sub) == 0 or plan not in s3:
                continue
            hp = None if plan == "BF" else sub["hyperparam"].to_numpy()
            pred = s3[plan].predict(
                stage3_feats(sub["n_pass"].to_numpy(), sub["rho"].to_numpy(),
                             sub["N"].to_numpy(), hp, sub["k"].to_numpy()))
            s3_true[plan]["yt"].append(sub["runtime_ms"].to_numpy())
            s3_true[plan]["yp"].append(pred)

        # per-plan predictions on the test pairs (vectorised per plan)
        te_ann = te_pp[te_pp["plan"].isin(ANN_PLANS)].copy()
        cols = {"P": np.zeros(len(te_ann)), "hp": np.zeros(len(te_ann)),
                "hpb": np.zeros(len(te_ann)), "cost": np.zeros(len(te_ann)),
                "costb": np.zeros(len(te_ann)), "rt": np.zeros(len(te_ann)),
                "rtb": np.zeros(len(te_ann)), "rec": np.zeros(len(te_ann)),
                "recb": np.zeros(len(te_ann)), "hp_uni": np.zeros(len(te_ann))}
        plan_arr = te_ann["plan"].to_numpy()
        for plan in ANN_PLANS:
            m = plan_arr == plan
            if not m.any():
                continue
            grid = GRID[plan]
            sub = te_ann[m]
            X = sub[STAGE12_FEATS].to_numpy()
            n_pass, rho, N = sub["n_pass"].to_numpy(), sub["rho"].to_numpy(), sub["N"].to_numpy()
            kk = sub["k"].to_numpy()
            P = feas_clf[plan].predict(X)
            hp = predict_hp(hp_reg[plan], X, grid)
            hpb = np.array([bump_one(v, grid) for v in hp])
            cost = s3[plan].predict(stage3_feats(n_pass, rho, N, hp, kk))
            costb = s3[plan].predict(stage3_feats(n_pass, rho, N, hpb, kk))
            keys = list(zip(sub["query_type"], sub["query_id_num"].astype(int),
                            sub["filter_id"].astype(int), sub["k"].astype(int)))
            rr = np.array([measure_at(meas, (*key, plan), h) for key, h in zip(keys, hp)])
            rrb = np.array([measure_at(meas, (*key, plan), h) for key, h in zip(keys, hpb)])
            uni = np.array([snap_ceil(v, grid) for v in hp_uni[plan].predict(X)])
            cols["P"][m], cols["hp"][m], cols["hpb"][m] = P, hp, hpb
            cols["cost"][m], cols["costb"][m] = cost, costb
            cols["rec"][m], cols["rt"][m] = rr[:, 0], rr[:, 1]
            cols["recb"][m], cols["rtb"][m] = rrb[:, 0], rrb[:, 1]
            cols["hp_uni"][m] = uni

            # --- Stage 1/2 diagnostics -----------------------------------
            feas_true = sub["feasible"].to_numpy()
            stage1[plan]["acc"].append(float(((P >= 0.5).astype(int) == feas_true).mean()))
            stage1[plan]["auc"].append(auc(feas_true, P))
            gated = P >= 0.5
            stage2[plan]["n"] += int(gated.sum())
            stage2[plan]["hit"] += int((rr[gated, 0] >= RECALL_TARGET).sum())
            for h, bt, ft in zip(hp[gated], sub["best_hp"].to_numpy()[gated], feas_true[gated]):
                if ft == 1:
                    ti, pi = grid_index(bt, grid), grid_index(h, grid)
                    if ti >= 0 and pi >= 0:
                        stage2[plan]["step"].append(pi - ti)
            # Stage 3 at predicted (ablation) hp on gated pairs
            s3_pred[plan]["yt"].extend(rr[gated, 1].tolist())
            s3_pred[plan]["yp"].extend(cost[gated].tolist())
            # unified-model feasibility/hit (reference)
            feas_uni = uni > 0
            stage2u[plan]["acc"].append(float((feas_uni.astype(int) == feas_true).mean()))
            stage2u[plan]["n"] += int(feas_uni.sum())
            if feas_uni.any():
                ru = np.array([measure_at(meas, (*key, plan), h)[0]
                               for key, h, fu in zip(keys, uni, feas_uni) if fu])
                stage2u[plan]["hit"] += int((ru >= RECALL_TARGET).sum())

        for c in cols:
            te_ann[c] = cols[c]

        # BF predicted cost per test pair (batch)
        te_bf = te_pp[te_pp["plan"] == "BF"].copy()
        te_bf["cost"] = s3["BF"].predict(
            stage3_feats(te_bf["n_pass"].to_numpy(), te_bf["rho"].to_numpy(),
                         te_bf["N"].to_numpy(), None, te_bf["k"].to_numpy()))
        bf_cost = {(r.query_type, int(r.query_id_num), int(r.filter_id), int(r.k)): float(r.cost)
                   for r in te_bf.itertuples()}

        # assemble per-pair optimizer records
        for pair, sub in te_ann.groupby(PAIR_KEYS):
            pair = (pair[0], int(pair[1]), int(pair[2]), int(pair[3]))
            bf_key = (*pair, "BF")
            bf_rec, bf_rt = measure_at(meas, bf_key, 0)
            cands = {"BF": {"p_feas": 1.0, "cost": bf_cost[pair], "cost_b": bf_cost[pair],
                            "rt": bf_rt, "rt_b": bf_rt, "rec": bf_rec, "rec_b": bf_rec}}
            hnsw = None
            for r in sub.itertuples():
                if r.plan == "HNSW-pre":
                    hnsw = {"rt": r.rt, "rec": r.rec, "rt_b": r.rtb, "rec_b": r.recb}
                if r.P >= 0.5:
                    cands[r.plan] = {"p_feas": float(r.P), "cost": r.cost, "cost_b": r.costb,
                                     "rt": r.rt, "rt_b": r.rtb, "rec": r.rec, "rec_b": r.recb}
            if hnsw is None:
                hnsw = {"rt": bf_rt, "rec": 1.0, "rt_b": bf_rt, "rec_b": 1.0}
            op, oc, orc = oracle[pair]
            records.append({"pair": pair, "cands": cands, "oracle_plan": op,
                            "oracle_cost": oc, "oracle_recall": orc,
                            "bf_rt": bf_rt, "hnsw": hnsw})

    diag = {"stage1": stage1, "stage2": stage2, "stage2u": stage2u,
            "s3_true": s3_true, "s3_pred": s3_pred}
    return records, diag


# --------------------------------------------------------------------------- #
# decision rule + optimizer scoring (cheap; sweep gamma/lambda/bump w/o refit)
# --------------------------------------------------------------------------- #
def decide(rec, gamma, lam, bump):
    ck, rk = ("cost_b", "rt_b") if bump else ("cost", "rt")
    reck = "rec_b" if bump else "rec"
    best_plan, best_score = "BF", np.inf
    for plan, c in rec["cands"].items():
        m = 1.0 if plan == "BF" else gamma * (1.0 + lam * (1.0 - c["p_feas"]))
        score = c[ck] * m
        if score < best_score:
            best_score, best_plan = score, plan
    c = rec["cands"][best_plan]
    return best_plan, c[rk], c[reck]


def optimizer_metrics(records, gamma, lam, bump=False):
    n = len(records)
    realised = np.empty(n)
    oracle = np.empty(n)
    bf = np.empty(n)
    hnsw = np.empty(n)
    recall = np.empty(n)
    is_oracle = np.zeros(n, bool)
    chosen = []
    hk = "rt_b" if bump else "rt"
    reck = "rec_b" if bump else "rec"
    for i, rec in enumerate(records):
        plan, rt, rcl = decide(rec, gamma, lam, bump)
        realised[i] = rt
        recall[i] = rcl
        oracle[i] = rec["oracle_cost"]
        bf[i] = rec["bf_rt"]
        hnsw[i] = rec["hnsw"][hk]
        is_oracle[i] = plan == rec["oracle_plan"]
        chosen.append(plan)
    fail = recall < RECALL_TARGET
    hnsw_rec = np.array([rec["hnsw"][reck] for rec in records])
    regret = realised / np.clip(oracle, 1e-9, None)
    tot_s = realised.sum() / 1000.0
    return {
        "gamma": gamma, "lam": lam, "bump": int(bump), "n": n,
        "avg_recall": float(recall.mean()),
        "recall_fail_rate": float(fail.mean()),
        "qps": float(n / tot_s),
        "chosen_is_oracle_pct": float(100 * is_oracle.mean()),
        "mean_regret": float(regret.mean()), "median_regret": float(np.median(regret)),
        "total_realised_ms": float(realised.sum()), "total_oracle_ms": float(oracle.sum()),
        "total_bf_ms": float(bf.sum()), "total_hnsw_ms": float(hnsw.sum()),
        "bf_qps": float(n / (bf.sum() / 1000.0)),
        "oracle_qps": float(n / (oracle.sum() / 1000.0)),
        "hnsw_qps": float(n / (hnsw.sum() / 1000.0)),
        "hnsw_avg_recall": float(hnsw_rec.mean()),
        "hnsw_fail_rate": float((hnsw_rec < RECALL_TARGET).mean()),
        "speedup_vs_bf": float(bf.sum() / realised.sum()),
        "speedup_vs_hnsw": float(hnsw.sum() / realised.sum()),
        "chosen": chosen, "realised": realised, "recall": recall, "fail": fail,
    }


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
        acc = np.nanmean(s1[p]["acc"]) if s1[p]["acc"] else np.nan
        au = np.nanmean(s1[p]["auc"]) if s1[p]["auc"] else np.nan
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


def report_optimizer(m, label):
    print(f"\n--- Optimizer [{label}]  (gamma={m['gamma']}, lambda={m['lam']}, "
          f"safety_bump={bool(m['bump'])}) ---")
    print(f"  AVERAGE recall of chosen plans : {m['avg_recall']:.4f}   "
          f"(recall<0.95 on {100*m['recall_fail_rate']:.1f}% of pairs)")
    print(f"  THROUGHPUT                     : {m['qps']:.1f} QPS  "
          f"(BF {m['bf_qps']:.1f}, oracle {m['oracle_qps']:.1f}, HNSW-pre {m['hnsw_qps']:.1f})")
    print(f"  chosen == oracle plan          : {m['chosen_is_oracle_pct']:.1f}%")
    print(f"  mean regret (realised/oracle)  : {m['mean_regret']:.3f}   median {m['median_regret']:.3f}")
    print(f"  vs always-BF                   : {m['speedup_vs_bf']:.2f}x throughput  (BF recall 1.000)")
    print(f"  vs always-HNSW-pre             : {m['speedup_vs_hnsw']:.2f}x throughput  "
          f"(HNSW-pre avg recall {m['hnsw_avg_recall']:.4f})")
    mix = pd.Series(m["chosen"]).value_counts()
    print("  chosen-plan mix: " + ", ".join(f"{p}={c}" for p, c in mix.items()))


def plot_pareto(pareto, out_png, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(pareto["avg_recall"], pareto["qps"], "o-", color="#5aa0ff")
    for _, r in pareto.iterrows():
        ax.annotate(f"g={r['gamma']:g}", (r["avg_recall"], r["qps"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(RECALL_TARGET, color="#e2585f", ls="--", lw=1, alpha=0.7,
               label=f"recall target {RECALL_TARGET}")
    ax.set_yscale("log")
    ax.set_xlabel("Average recall of chosen plans")
    ax.set_ylabel("Throughput [QPS, log]")
    ax.set_title(f"gamma-sweep Pareto (throughput vs average recall) -- {label}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved Pareto plot -> {out_png}")


def plot_benefits(records, ops, pareto, out_png, label):
    """Two panels: (A) total-latency bars vs baselines + oracle, each annotated
    with its recall-miss rate; (B) the optimizer's latency-vs-recall-safety
    frontier with the always-BF / always-HNSW-pre / oracle reference points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    bf_qps = n / (sum(r["bf_rt"] for r in records) / 1000.0)
    oracle_qps = n / (sum(r["oracle_cost"] for r in records) / 1000.0)
    hnsw_qps = n / (sum(r["hnsw"]["rt"] for r in records) / 1000.0)
    oracle_rec = float(np.mean([r["oracle_recall"] for r in records]))
    hnsw_rec = float(np.mean([r["hnsw"]["rec"] for r in records]))
    d, lb = ops["default"], ops["+lambda+bump"]

    bars = [  # (label, qps, avg_recall, color)
        ("always-BF", bf_qps, 1.0, "#e2585f"),
        ("oracle", oracle_qps, oracle_rec, "#39c07a"),
        ("always\nHNSW-pre", hnsw_qps, hnsw_rec, "#e6a23c"),
        (f"optimizer\n(g={d['gamma']:g})", d["qps"], d["avg_recall"], "#5aa0ff"),
        ("optimizer\n(+lam+bump)", lb["qps"], lb["avg_recall"], "#b07ad6"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A -- throughput bars (log) labelled with average recall
    x = np.arange(len(bars))
    vals = [b[1] for b in bars]
    ax1.bar(x, vals, color=[b[3] for b in bars])
    ax1.set_yscale("log")
    ax1.set_ylim(top=max(vals) * 3)
    ax1.set_ylabel("Throughput [QPS, log]")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax1.set_title(f"Throughput over {n} (query,filter) pairs -- {label}")
    for xi, b in zip(x, bars):
        ax1.text(xi, b[1] * 1.05, f"{b[1]:.0f} QPS\nrecall {b[2]:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.text(0.02, 0.97,
             f"optimizer (g={d['gamma']:g}) = {d['speedup_vs_bf']:.0f}x throughput of always-BF\n"
             f"avg recall {d['avg_recall']:.3f}; chosen == oracle plan {d['chosen_is_oracle_pct']:.0f}%",
             transform=ax1.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round", fc="#f4f7ff", ec="#c7d6f0"))

    # Panel B -- throughput vs average-recall frontier + reference points
    ax2.plot(pareto["avg_recall"], pareto["qps"], "o-", color="#5aa0ff",
             label="optimizer (gamma-sweep)", zorder=3)
    for _, r in pareto.iterrows():
        ax2.annotate(f"g={r['gamma']:g}", (r["avg_recall"], r["qps"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax2.scatter([1.0], [bf_qps], marker="s", s=110, color="#e2585f",
                label="always-BF (recall 1.0)", zorder=4)
    ax2.scatter([oracle_rec], [oracle_qps], marker="D", s=110, color="#39c07a",
                label="oracle", zorder=4)
    ax2.scatter([hnsw_rec], [hnsw_qps], marker="^", s=110, color="#e6a23c",
                label="always-HNSW-pre", zorder=4)
    ax2.scatter([lb["avg_recall"]], [lb["qps"]], marker="*", s=230,
                color="#b07ad6", label="optimizer +lam+bump", zorder=5)
    ax2.axvline(RECALL_TARGET, color="#888", ls="--", lw=1, alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_xlabel("Average recall of chosen plans")
    ax2.set_ylabel("Throughput [QPS, log]")
    ax2.set_title(f"Throughput vs average recall -- {label}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benefits plot -> {out_png}")


def plot_decision_scatter(records, chosen, feat, out_png, label, gamma):
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

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    for j, qt in enumerate(["movies", "reviews"]):
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
    fig.suptitle(f"Logical-plan decisions across (selectivity, rho) -- {label} GLS",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved decision scatter -> {out_png}")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_one(path, gls_label, out, gamma, lam, min_leaf):
    print(f"\n{'='*74}\n GLS = {gls_label}   ({path})\n{'='*74}")
    sweep = load_sweep(path)
    pp = build_pairplan(sweep)
    meas = build_measure(sweep)
    pairs = [tuple(x) for x in pp[PAIR_KEYS].drop_duplicates().to_numpy()]
    pairs = [(p[0], int(p[1]), int(p[2]), int(p[3])) for p in pairs]
    oracle = build_oracle(meas, pairs)
    print(f"  {len(pairs)} (query,filter) pairs; plans: {sorted(pp['plan'].unique())}")
    print("  per-plan feasibility base rate (fraction reaching recall>=0.95):")
    base = pp.groupby("plan")["feasible"].mean().reindex(ALL_PLANS)
    print("    " + "  ".join(f"{p}={v:.3f}" for p, v in base.items()))

    feat = {}
    for pair, g in pp.groupby(PAIR_KEYS):
        r0 = g.iloc[0]
        feat[(pair[0], int(pair[1]), int(pair[2]), int(pair[3]))] = (
            float(r0["sigma"]), float(r0["rho"]), pair[0])

    records, diag = run_cv(sweep, pp, meas, oracle, min_leaf=min_leaf)

    pd.DataFrame(report_stages(diag)).to_csv(out / f"stage12_metrics_{gls_label}.csv", index=False)
    pd.DataFrame(report_s3(diag)).to_csv(out / f"stage3_metrics_{gls_label}.csv", index=False)

    # operating points: default / +lambda / +safety-bump / +both
    print("\n--- Operating points (gamma={:g}) ---".format(gamma))
    ops = {
        "default": optimizer_metrics(records, gamma, 0.0, False),
        "+lambda": optimizer_metrics(records, gamma, lam, False),
        "+safety_bump": optimizer_metrics(records, gamma, 0.0, True),
        "+lambda+bump": optimizer_metrics(records, gamma, lam, True),
    }
    for name, m in ops.items():
        report_optimizer(m, f"{gls_label} / {name}")
    knob_rows = [{"operating_point": name, **{k: m[k] for k in (
        "gamma", "lam", "bump", "recall_fail_rate", "chosen_is_oracle_pct",
        "mean_regret", "total_realised_ms", "speedup_vs_bf", "speedup_vs_hnsw")}}
        for name, m in ops.items()]
    pd.DataFrame(knob_rows).to_csv(out / f"operating_points_{gls_label}.csv", index=False)

    # per-pair selection dump at the default operating point
    m = ops["default"]
    sel = pd.DataFrame({
        "query_type": [r["pair"][0] for r in records],
        "query_id_num": [r["pair"][1] for r in records],
        "filter_id": [r["pair"][2] for r in records],
        "k": [r["pair"][3] for r in records],
        "sigma": [feat[r["pair"]][0] for r in records],
        "rho": [feat[r["pair"]][1] for r in records],
        "chosen": m["chosen"], "realised_ms": m["realised"],
        "recall": m["recall"], "recall_fail": m["fail"],
        "oracle_plan": [r["oracle_plan"] for r in records],
        "oracle_ms": [r["oracle_cost"] for r in records],
        "bf_ms": [r["bf_rt"] for r in records],
    })
    sel.to_csv(out / f"selection_{gls_label}.csv", index=False)

    # per-dataset breakdown at the default operating point
    sel["is_oracle"] = [c == r["oracle_plan"] for c, r in zip(m["chosen"], records)]
    print(f"\n--- Per-dataset breakdown [{gls_label}, default gamma={gamma:g}] ---")
    print(f"  {'dataset':8s} {'n':>5s} {'QPS':>7s} {'BF_QPS':>7s} "
          f"{'speedup':>7s} {'avg_recall':>10s} {'miss':>6s} {'==oracle':>8s}")
    for ds, g in sel.groupby("query_type"):
        qps = len(g) / (g["realised_ms"].sum() / 1000)
        bf_qps = len(g) / (g["bf_ms"].sum() / 1000)
        print(f"  {ds:8s} {len(g):5d} {qps:7.1f} {bf_qps:7.1f} {qps/bf_qps:6.1f}x "
              f"{g['recall'].mean():10.4f} {100*g['recall_fail'].mean():5.1f}% "
              f"{100*g['is_oracle'].mean():7.1f}%")
    plot_decision_scatter(records, m["chosen"], feat,
                          out / f"decision_scatter_{gls_label}.png", gls_label, gamma)

    # gamma-sweep Pareto (lambda=0, no bump)
    pareto = pd.DataFrame([{k: mm[k] for k in (
        "gamma", "lam", "bump", "avg_recall", "qps", "recall_fail_rate",
        "chosen_is_oracle_pct", "mean_regret", "median_regret", "total_realised_ms",
        "total_oracle_ms", "total_bf_ms", "total_hnsw_ms", "speedup_vs_bf",
        "speedup_vs_hnsw")}
        for mm in (optimizer_metrics(records, g, 0.0, False) for g in GAMMA_SWEEP)])
    pareto.to_csv(out / f"gamma_pareto_{gls_label}.csv", index=False)
    print(f"\n--- gamma-sweep Pareto [{gls_label}] (throughput vs average recall) ---")
    print(pareto[["gamma", "avg_recall", "qps", "recall_fail_rate",
                  "speedup_vs_bf", "mean_regret"]].to_string(
        index=False, formatters={
            "avg_recall": lambda x: f"{x:.4f}",
            "qps": lambda x: f"{x:.1f}",
            "recall_fail_rate": lambda x: f"{100*x:.1f}%",
            "speedup_vs_bf": lambda x: f"{x:.2f}x",
            "mean_regret": lambda x: f"{x:.3f}"}))
    plot_pareto(pareto, out / f"gamma_pareto_{gls_label}.png", gls_label)
    plot_benefits(records, ops, pareto, out / f"benefits_{gls_label}.png", gls_label)
    return ops["default"]


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
    ap.add_argument("--out", default=str(root / "plots/qo_prototype"))
    args = ap.parse_args()

    exact_path = Path(args.results)
    est_path = exact_path.parent / "gls_est" / "all_query_results.csv"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.gls in ("exact", "both"):
        targets.append(("exact", exact_path))
    if args.gls in ("estimated", "both"):
        if est_path.is_file():
            targets.append(("estimated", est_path))
        elif args.gls == "estimated":
            raise SystemExit(f"estimated GLS CSV not found: {est_path}")
        else:
            print(f"(skipping estimated GLS: {est_path} not found)")

    summary = {}
    for label, path in targets:
        summary[label] = run_one(path, label, out, args.gamma, args.lam, args.min_leaf)

    if len(summary) == 2:
        e, s = summary["exact"], summary["estimated"]
        print(f"\n{'='*74}\n exact vs estimated GLS  (confirm estimated ~= exact)\n{'='*74}")
        print(f"  average recall      : exact {e['avg_recall']:.4f}   "
              f"estimated {s['avg_recall']:.4f}")
        print(f"  throughput [QPS]    : exact {e['qps']:.1f}   estimated {s['qps']:.1f}")
        print(f"  speedup vs BF       : exact {e['speedup_vs_bf']:.2f}x   "
              f"estimated {s['speedup_vs_bf']:.2f}x")
        print(f"  chosen==oracle      : exact {e['chosen_is_oracle_pct']:.1f}%   "
              f"estimated {s['chosen_is_oracle_pct']:.1f}%")

    print(f"\nOutputs -> {out}")


if __name__ == "__main__":
    main()
