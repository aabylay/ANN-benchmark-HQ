#!/usr/bin/env python3
"""FANNS query optimizer -- RIGOROUS BENEFIT ANALYSIS (v2).

Companion to analysis/qo_prototype.py (design: prompts/qo_prototype.txt). This
module answers a sharper question than the prototype's headline: *when* and *how
much* does the per-query optimizer actually beat simple STATIC policies, measured
AT MATCHED RECALL -- not against the misleading "always-HNSW-pre @ our own
predicted per-query hp" baseline (which nearly coincides with the optimizer
because the optimizer picks HNSW-pre ~66% of the time with the same hp).

What this adds on top of qo_prototype.py:

  * A true "NO param-choosing" STATIC baseline: for HNSW-pre and IVF-pre, apply
    ONE fixed hp to EVERY query and sweep that fixed hp -> the plan's STATIC
    FRONTIER (throughput vs average recall, plus the per-query recall
    distribution). static-best = upper envelope of the two.
  * MATCHED-RECALL comparison: optimizer QPS gain over static-best AT EQUAL
    average recall and at a matched per-query recall FLOOR (p10 >= 0.90), with the
    gain decomposed into plan-routing vs cheaper-hp.
  * Realised per-query recall DISTRIBUTIONS (CDF, tail fractions, p1), decision
    quality (==oracle, regret, confusion, win/loss-vs-static), per-dataset and
    per-selectivity breakdowns, model ablations (features / gate / bump / lambda /
    Stage-3 pooling / CART-vs-sklearn), recall-target sensitivity, and a
    cost-model-honesty accounting (verify + fallback-to-BF upper bound on the true
    cost of silent recall misses, vs an online hp ramp).

Everything is grouped 5-fold CV splitting *queries*; runs on exact and estimated
GLS. Reuses the numpy CART / LogOLS / loaders from qo_prototype.py; sklearn is
OPTIONAL (used only for the CART-vs-RandomForest ablation and if --model asks).
Outputs -> analysis/plots/qo_prototype/ with a  full_*  prefix so it never
clobbers the prototype's own artefacts; also writes FINDINGS.md.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import qo_prototype as qo
from qo_prototype import (ANN_PLANS, ALL_PLANS, GRID, HNSW_GRID, IVF_GRID,
                          N_TABLE, PAIR_KEYS, PLAN_PLOT_COLORS, CART, LogOLS,
                          load_sweep)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # sklearn/lightgbm feature-name spam

RT_DEFAULT = 0.95
# selectivity ranges 0.005..1.0 in the data; edges chosen so every bin is populated
SEL_BINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.01]
SEL_LABELS = ["<2%", "2-5%", "5-10%", "10-20%", "20-50%", ">50%"]

try:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAVE_LGBM = True
except Exception:  # pragma: no cover
    HAVE_LGBM = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False

# backends available for the headline optimizer / ablation, in preference order
BACKENDS = (["cart"] + (["sk-tree", "sk-rf"] if HAVE_SKLEARN else [])
            + (["lgbm"] if HAVE_LGBM else []) + (["xgb"] if HAVE_XGB else []))


# --------------------------------------------------------------------------- #
# pluggable model backend (numpy CART default; sklearn tree / RF optional)
# --------------------------------------------------------------------------- #
class _SkProba:
    """Wrap an sklearn classifier so .predict returns P(feasible)."""

    def __init__(self, est):
        self.est = est
        self._const = None

    def fit(self, X, y):
        y = np.asarray(y, float)
        if len(np.unique(y)) < 2:
            self._const = float(y[0]) if len(y) else 0.0
        else:
            self.est.fit(X, y)
        return self

    def predict(self, X):
        if self._const is not None:
            return np.full(len(X), self._const)
        return self.est.predict_proba(X)[:, 1]


class _SkReg:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, np.asarray(y, float))
        return self

    def predict(self, X):
        return self.est.predict(X)


def make_clf(backend, min_leaf, max_depth=3):
    if backend == "cart":
        return CART("clf", max_depth=max_depth, min_leaf=min_leaf)
    if backend == "sk-tree":
        return _SkProba(DecisionTreeClassifier(max_depth=max_depth,
                                               min_samples_leaf=min_leaf))
    if backend == "sk-rf":
        return _SkProba(RandomForestClassifier(n_estimators=300, max_depth=None,
                                               min_samples_leaf=min_leaf, n_jobs=-1,
                                               random_state=0))
    if backend == "lgbm":
        return _SkProba(LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                       num_leaves=15, max_depth=4,
                                       min_child_samples=min_leaf, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=1.0,
                                       n_jobs=-1, random_state=0, verbosity=-1))
    if backend == "xgb":
        return _SkProba(XGBClassifier(n_estimators=400, learning_rate=0.05,
                                      max_depth=4, min_child_weight=float(min_leaf) / 3,
                                      subsample=0.8, colsample_bytree=1.0, n_jobs=-1,
                                      random_state=0, verbosity=0, eval_metric="logloss"))
    raise ValueError(backend)


def make_reg(backend, min_leaf, max_depth=3):
    if backend == "cart":
        return CART("reg", max_depth=max_depth, min_leaf=min_leaf)
    if backend == "sk-tree":
        return _SkReg(DecisionTreeRegressor(max_depth=max_depth,
                                            min_samples_leaf=min_leaf))
    if backend == "sk-rf":
        return _SkReg(RandomForestRegressor(n_estimators=300, max_depth=None,
                                            min_samples_leaf=min_leaf, n_jobs=-1,
                                            random_state=0))
    if backend == "lgbm":
        return _SkReg(LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                    num_leaves=15, max_depth=4,
                                    min_child_samples=min_leaf, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=1.0,
                                    n_jobs=-1, random_state=0, verbosity=-1))
    if backend == "xgb":
        return _SkReg(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                   min_child_weight=float(min_leaf) / 3, subsample=0.8,
                                   colsample_bytree=1.0, n_jobs=-1, random_state=0,
                                   verbosity=0))
    raise ValueError(backend)


# --------------------------------------------------------------------------- #
# ground-truth structure (built once per GLS source)
# --------------------------------------------------------------------------- #
def build_gt(sweep: pd.DataFrame):
    """pair -> features + per-plan (hps, recall, runtime_ms) arrays (sorted hp)."""
    gt = {}
    for (qt, qid, fid), g in sweep.groupby(PAIR_KEYS, sort=False):
        r0 = g.iloc[0]
        pair = (qt, int(qid), int(fid))
        d = {"qt": qt, "sigma": float(r0["sigma"]), "rho": float(r0["rho"]),
             "N": float(r0["N"]), "n_pass": float(r0["n_pass"]), "plans": {}}
        for plan, gp in g.groupby("plan", sort=False):
            gp = gp.sort_values("hyperparam")
            d["plans"][plan] = (gp["hyperparam"].to_numpy(float),
                                gp["recall"].to_numpy(float),
                                gp["runtime_ms"].to_numpy(float))
        gt[pair] = d
    return gt


def dataset_grid(gt, pairs, plan):
    """hps common to EVERY pair of this scope for this plan (intersection)."""
    common = None
    for p in pairs:
        pl = gt[p]["plans"].get(plan)
        if pl is None:
            continue
        s = set(pl[0].tolist())
        common = s if common is None else (common & s)
    return sorted(common) if common else []


def feas_besthp(hps, rec, target):
    ok = np.where(rec >= target)[0]
    if len(ok) == 0:
        return 0, 0.0
    return 1, float(hps[ok[0]])


def snap_ceil_grid(raw, grid):
    for v in grid:
        if v >= raw:
            return v
    return grid[-1]


# --------------------------------------------------------------------------- #
# ground-truth policy frontiers (NOT learned -- fixed policies / bounds)
# --------------------------------------------------------------------------- #
def _dist(recs, rts):
    recs = np.asarray(recs, float)
    rts = np.asarray(rts, float)
    return {
        "n": len(recs), "avg_recall": float(recs.mean()),
        "qps": float(len(recs) / (rts.sum() / 1000.0)),
        "tot_s": float(rts.sum() / 1000.0),
        "p1": float(np.percentile(recs, 1)), "p10": float(np.percentile(recs, 10)),
        "p50": float(np.percentile(recs, 50)),
        "frac_ge_95": float((recs >= 0.95).mean()),
        "frac_lt_95": float((recs < 0.95).mean()),
        "frac_lt_90": float((recs < 0.90).mean()),
        "frac_lt_80": float((recs < 0.80).mean()),
    }


def static_frontier(gt, pairs, plan):
    """Fixed-hp STATIC policy: apply ONE hp to every pair; sweep hp over the grid.

    Only hps present for EVERY pair in scope are used (a true 'one fixed hp for
    everyone' policy). Returns one point per fixed hp.
    """
    grid = dataset_grid(gt, pairs, plan)
    rows = []
    for hp in grid:
        recs, rts = [], []
        for p in pairs:
            pl = gt[p]["plans"].get(plan)
            if pl is None:
                continue
            hps, rc, rt = pl
            j = np.where(hps == hp)[0]
            if len(j):
                recs.append(rc[j[0]])
                rts.append(rt[j[0]])
        d = _dist(recs, rts)
        d.update({"plan": plan, "hp": hp, "policy": "static"})
        rows.append(d)
    return rows


def single_plan_hp_frontier(gt, pairs, plan, targets):
    """SINGLE plan, per-query ORACLE min-hp reaching a swept target t (else max hp).

    Isolates the *cheaper-hp* lever (no plan routing): an upper bound on what hp
    tuning alone can buy on one plan.
    """
    rows = []
    for t in targets:
        recs, rts = [], []
        for p in pairs:
            pl = gt[p]["plans"].get(plan)
            if pl is None:
                continue
            hps, rc, rt = pl
            ok = np.where(rc >= t)[0]
            j = ok[0] if len(ok) else len(hps) - 1
            recs.append(rc[j])
            rts.append(rt[j])
        d = _dist(recs, rts)
        d.update({"plan": plan, "target": t, "policy": "single_plan_hp"})
        rows.append(d)
    return rows


def oracle_route_frontier(gt, pairs, targets, plans=ALL_PLANS):
    """Per-query cheapest plan+hp reaching a swept target t (BF always qualifies).

    routing + hp upper bound. At t=target this is the standard oracle.
    """
    rows = []
    for t in targets:
        recs, rts, mix = [], [], {}
        for p in pairs:
            best_c, best_r, best_p = np.inf, 1.0, "BF"
            for plan in plans:
                pl = gt[p]["plans"].get(plan)
                if pl is None:
                    continue
                hps, rc, rt = pl
                if plan == "BF":
                    c, r = rt[0], rc[0]
                else:
                    ok = np.where(rc >= t)[0]
                    if len(ok) == 0:
                        continue
                    c, r = rt[ok[0]], rc[ok[0]]
                if c < best_c:
                    best_c, best_r, best_p = c, r, plan
            recs.append(best_r)
            rts.append(best_c)
            mix[best_p] = mix.get(best_p, 0) + 1
        d = _dist(recs, rts)
        d.update({"target": t, "policy": "oracle_route", "mix": mix})
        rows.append(d)
    return rows


def pareto_envelope(points):
    """Non-dominated (higher recall & higher qps better); sorted by recall asc."""
    pts = sorted(points, key=lambda d: (d["avg_recall"], d["qps"]))
    env, best_q = [], -np.inf
    for d in reversed(pts):  # high recall -> low; keep increasing qps
        if d["qps"] > best_q:
            env.append(d)
            best_q = d["qps"]
    return list(reversed(env))


def interp_qps(points, x):
    """log-linear interp of qps at avg_recall=x on a frontier; None if out of range."""
    pts = sorted(points, key=lambda d: d["avg_recall"])
    xs = [d["avg_recall"] for d in pts]
    ys = [d["qps"] for d in pts]
    if x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:
        return None
    for i in range(len(xs) - 1):
        if xs[i] - 1e-9 <= x <= xs[i + 1] + 1e-9:
            if xs[i + 1] == xs[i]:
                return max(ys[i], ys[i + 1])
            f = (x - xs[i]) / (xs[i + 1] - xs[i])
            return float(np.exp(np.log(ys[i]) + f * (np.log(ys[i + 1]) - np.log(ys[i]))))
    return ys[-1]


# --------------------------------------------------------------------------- #
# learned optimizer -- CV producing rich per-pair records
# --------------------------------------------------------------------------- #
def assign_folds(pairs, n_folds=5, seed=0):
    groups = sorted({(p[0], p[1]) for p in pairs})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    fold_of = {groups[perm[i]]: i % n_folds for i in range(len(groups))}
    return {p: fold_of[(p[0], p[1])] for p in pairs}


def feat_s12(rows, use_logn):
    cols = [rows["sigma"], rows["rho"]]
    if use_logn:
        cols.append(np.log(rows["n_pass"]))
    return np.column_stack(cols)


def stage3_feats(n_pass, rho, N, hp, use_logn):
    cols = [np.log(n_pass), np.asarray(rho, float)]
    if use_logn:
        cols.append(np.log(N))
    cols.append(np.log(np.clip(np.asarray(hp, float), 1.0, None)))
    return np.column_stack(cols)


def pairplan_df(gt, pairs, target):
    rows = []
    for p in pairs:
        d = gt[p]
        for plan in ANN_PLANS:
            pl = d["plans"].get(plan)
            if pl is None:
                continue
            f, bh = feas_besthp(pl[0], pl[1], target)
            rows.append({"pair": p, "qt": p[0], "sigma": d["sigma"], "rho": d["rho"],
                         "N": d["N"], "n_pass": d["n_pass"], "plan": plan,
                         "feasible": f, "best_hp": bh})
    return pd.DataFrame(rows)


def sweep_rows(gt, pairs):
    """Long ground-truth rows for Stage-3 training (all plans, all hp)."""
    rows = []
    for p in pairs:
        d = gt[p]
        for plan, (hps, rc, rt) in d["plans"].items():
            for h, r, t in zip(hps, rc, rt):
                rows.append({"qt": p[0], "plan": plan, "n_pass": d["n_pass"],
                             "rho": d["rho"], "N": d["N"], "hp": h, "rt_ms": t})
    return pd.DataFrame(rows)


def fit_fold(gt, tr_pairs, target, cfg):
    """Fit per-plan feasibility clf + hp regressor (gated or unified) + Stage-3."""
    ppdf = pairplan_df(gt, tr_pairs, target)
    swr = sweep_rows(gt, tr_pairs)
    ml, bk = cfg["min_leaf"], cfg["backend"]
    md, logn = cfg.get("max_depth_s12", 3), cfg["use_logn_s12"]
    feas, hpreg, hpuni, s3 = {}, {}, {}, {}

    def _fit_s12(tr_plan):
        fc = make_clf(bk, ml, md).fit(feat_s12(tr_plan, logn), tr_plan["feasible"].to_numpy())
        hu = make_reg(bk, ml, md).fit(feat_s12(tr_plan, logn), tr_plan["best_hp"].to_numpy())
        fr = tr_plan[tr_plan["feasible"] == 1]
        if len(fr) >= 2 * ml:
            hr = make_reg(bk, ml, md).fit(feat_s12(fr, logn), np.log(fr["best_hp"].to_numpy()))
        else:
            hr = float(np.log(GRID[plan][-1]))
        return fc, hu, hr

    for plan in ANN_PLANS:
        tr = ppdf[ppdf["plan"] == plan]
        if cfg.get("per_dataset_s12"):
            feas[plan], hpuni[plan], hpreg[plan] = {}, {}, {}
            for ds, tds in tr.groupby("qt"):
                feas[plan][ds], hpuni[plan][ds], hpreg[plan][ds] = _fit_s12(tds)
        else:
            feas[plan], hpuni[plan], hpreg[plan] = _fit_s12(tr)
    # Stage 3: pooled (+logN) or per-dataset (no logN, one model per table)
    for plan in ALL_PLANS:
        s = swr[swr["plan"] == plan]
        if len(s) == 0:
            continue
        if cfg["s3_per_dataset"]:
            s3[plan] = {}
            for ds, sg in s.groupby("qt"):
                s3[plan][ds] = LogOLS().fit(
                    stage3_feats(sg["n_pass"].to_numpy(), sg["rho"].to_numpy(),
                                 sg["N"].to_numpy(), sg["hp"].to_numpy(), False),
                    sg["rt_ms"].to_numpy())
        else:
            s3[plan] = LogOLS().fit(
                stage3_feats(s["n_pass"].to_numpy(), s["rho"].to_numpy(),
                             s["N"].to_numpy(), s["hp"].to_numpy(), True),
                s["rt_ms"].to_numpy())
    return {"feas": feas, "hpreg": hpreg, "hpuni": hpuni, "s3": s3, "cfg": cfg}


def _s3_cost(model, cfg, n_pass, rho, N, hp, ds):
    if cfg["s3_per_dataset"]:
        return model[ds].predict(stage3_feats(n_pass, rho, N, hp, False))
    return model.predict(stage3_feats(n_pass, rho, N, hp, True))


def cv_records(gt, pairs, target, cfg, n_folds=5, seed=0):
    """Grouped-by-query CV. Returns per-pair records with ground-truth arrays and
    the learned predictions (P_feasible, predicted min-hp index, predicted cost per
    available hp) needed to score any (gamma, lambda, safety-step) policy offline.
    Also returns Stage-1/2/3 diagnostics."""
    fold_of = assign_folds(pairs, n_folds, seed)
    rec_by_pair = {}
    diag = {"s1_acc": {p: [] for p in ANN_PLANS}, "s1_auc": {p: [] for p in ANN_PLANS},
            "s2_hit": {p: [0, 0] for p in ANN_PLANS},
            "s2u_acc": {p: [] for p in ANN_PLANS}, "s2u_hit": {p: [0, 0] for p in ANN_PLANS},
            "s3t": {p: [[], []] for p in ALL_PLANS}, "s3p": {p: [[], []] for p in ANN_PLANS}}

    for fold in range(n_folds):
        tr_pairs = [p for p in pairs if fold_of[p] != fold]
        te_pairs = [p for p in pairs if fold_of[p] == fold]
        M = fit_fold(gt, tr_pairs, target, cfg)

        # initialise records for test pairs
        for p in te_pairs:
            d = gt[p]
            r = {"pair": p, "qt": p[0], "sigma": d["sigma"], "rho": d["rho"],
                 "N": d["N"], "n_pass": d["n_pass"], "fold": fold, "ann": {}}
            pl = d["plans"].get("BF")
            r["bf"] = {"rt": float(pl[2][0]), "rec": float(pl[1][0])}
            rec_by_pair[p] = r

        # per (plan, dataset) batched predictions
        for plan in ANN_PLANS:
            for ds in ("movies", "reviews"):
                sub = [p for p in te_pairs if p[0] == ds and plan in gt[p]["plans"]]
                if not sub:
                    continue
                grid = dataset_grid(gt, [p for p in tr_pairs if p[0] == ds] or sub, plan)
                if not grid:
                    grid = dataset_grid(gt, sub, plan)
                sig = np.array([gt[p]["sigma"] for p in sub])
                rho = np.array([gt[p]["rho"] for p in sub])
                npass = np.array([gt[p]["n_pass"] for p in sub])
                N = np.array([gt[p]["N"] for p in sub])
                Xrows = {"sigma": sig, "rho": rho, "n_pass": npass}
                X = feat_s12(Xrows, cfg["use_logn_s12"])
                pds = cfg.get("per_dataset_s12")
                feas_m = M["feas"][plan][ds] if pds else M["feas"][plan]
                hpm = M["hpreg"][plan][ds] if pds else M["hpreg"][plan]
                uni_m = M["hpuni"][plan][ds] if pds else M["hpuni"][plan]
                P = feas_m.predict(X)
                raw = (np.full(len(sub), np.exp(hpm)) if isinstance(hpm, float)
                       else np.exp(hpm.predict(X)))
                # predicted cost per grid hp (batched)
                cost_cols = {}
                for v in grid:
                    cost_cols[v] = _s3_cost(M["s3"][plan], cfg, npass, rho, N,
                                            np.full(len(sub), v), ds)
                uni_raw = uni_m.predict(X)
                for i, p in enumerate(sub):
                    hps, rc, rt = gt[p]["plans"][plan]
                    # align cost to the pair's own hps (use grid values present)
                    cost = np.array([cost_cols[h][i] if h in cost_cols
                                     else _s3_cost(M["s3"][plan], cfg,
                                                   npass[i:i+1], rho[i:i+1], N[i:i+1],
                                                   np.array([h]), ds)[0]
                                     for h in hps])
                    snap = snap_ceil_grid(raw[i], list(hps))
                    min_i = int(np.where(hps == snap)[0][0])
                    uni_snap = snap_ceil_grid(uni_raw[i], list(hps))
                    uni_i = int(np.where(hps == uni_snap)[0][0])
                    rec_by_pair[p]["ann"][plan] = {
                        "hps": hps, "rec": rc, "rt": rt, "cost": cost,
                        "P": float(P[i]), "min_i": min_i,
                        "uni_hp": (uni_snap if uni_raw[i] >= 0.5 * hps[0] else 0.0),
                        "uni_i": uni_i}

                # ----- diagnostics -----
                feas_true = np.array([feas_besthp(*gt[p]["plans"][plan][:2], target)[0]
                                      for p in sub])
                best_hp_true = np.array([feas_besthp(*gt[p]["plans"][plan][:2], target)[1]
                                         for p in sub])
                diag["s1_acc"][plan].append(float(((P >= 0.5).astype(int) == feas_true).mean()))
                diag["s1_auc"][plan].append(qo.auc(feas_true, P))
                gated = P >= 0.5
                for i, p in enumerate(sub):
                    if not gated[i]:
                        continue
                    mi = rec_by_pair[p]["ann"][plan]["min_i"]
                    hit = gt[p]["plans"][plan][1][mi] >= target
                    diag["s2_hit"][plan][0] += int(hit)
                    diag["s2_hit"][plan][1] += 1
                    diag["s3p"][plan][0].append(gt[p]["plans"][plan][2][mi])
                    diag["s3p"][plan][1].append(rec_by_pair[p]["ann"][plan]["cost"][mi])
                uni_feas = np.array([rec_by_pair[p]["ann"][plan]["uni_hp"] > 0 for p in sub])
                diag["s2u_acc"][plan].append(float((uni_feas.astype(int) == feas_true).mean()))
                for i, p in enumerate(sub):
                    if uni_feas[i]:
                        ui = rec_by_pair[p]["ann"][plan]["uni_i"]
                        diag["s2u_hit"][plan][0] += int(gt[p]["plans"][plan][1][ui] >= target)
                        diag["s2u_hit"][plan][1] += 1

        # Stage-3 @ true hp over full test sweep
        for plan in ALL_PLANS:
            sub = [p for p in te_pairs if plan in gt[p]["plans"]]
            for p in sub:
                hps, rc, rt = gt[p]["plans"][plan]
                ds = p[0]
                pred = _s3_cost(M["s3"][plan], cfg, np.full(len(hps), gt[p]["n_pass"]),
                                np.full(len(hps), gt[p]["rho"]),
                                np.full(len(hps), gt[p]["N"]), hps, ds)
                diag["s3t"][plan][0].extend(rt.tolist())
                diag["s3t"][plan][1].extend(pred.tolist())

    records = [rec_by_pair[p] for p in pairs]

    # oracle per pair (cheapest feasible plan+hp at target)
    for r in records:
        p = r["pair"]
        best_c, best_r, best_p = np.inf, 1.0, "BF"
        for plan in ALL_PLANS:
            pl = gt[p]["plans"].get(plan)
            if pl is None:
                continue
            hps, rc, rt = pl
            if plan == "BF":
                c, rr = rt[0], rc[0]
            else:
                ok = np.where(rc >= target)[0]
                if len(ok) == 0:
                    continue
                c, rr = rt[ok[0]], rc[ok[0]]
            if c < best_c:
                best_c, best_r, best_p = c, rr, plan
        r["oracle"] = (best_p, float(best_c), float(best_r))
    return records, diag


# --------------------------------------------------------------------------- #
# policy evaluation on records
# --------------------------------------------------------------------------- #
def eval_policy(records, gamma=1.0, lam=0.0, steps=0):
    """Full optimizer decision. steps>=big => clamp to max hp (safety)."""
    n = len(records)
    chosen = np.empty(n, object)
    rt = np.empty(n)
    rec = np.empty(n)
    isor = np.zeros(n, bool)
    for i, r in enumerate(records):
        best_plan, best_score, best_i = "BF", r["bf"]["rt"] * 1.0, 0
        # NB: decision uses PREDICTED cost; BF predicted cost approximated by rt is
        # unfair to BF -> use its Stage-3 pred if present, else measured rt.
        best_score = r.get("bf_cost", r["bf"]["rt"])
        for plan, d in r["ann"].items():
            if d["P"] < 0.5:
                continue
            j = min(d["min_i"] + steps, len(d["hps"]) - 1)
            score = d["cost"][j] * gamma * (1.0 + lam * (1.0 - d["P"]))
            if score < best_score:
                best_score, best_plan, best_i = score, plan, j
        if best_plan == "BF":
            rt[i], rec[i] = r["bf"]["rt"], r["bf"]["rec"]
        else:
            d = r["ann"][best_plan]
            rt[i], rec[i] = d["rt"][best_i], d["rec"][best_i]
        chosen[i] = best_plan
        isor[i] = best_plan == r["oracle"][0]
    return {"chosen": chosen, "rt": rt, "rec": rec, "is_oracle": isor}


def eval_single_plan(records, plan, steps=0, predicted=True):
    """Always run `plan` at its per-query predicted min-hp (+steps). If predicted=
    False, use ORACLE min-hp for the pair's target (upper bound). Ignores gating."""
    n = len(records)
    rt = np.empty(n)
    rec = np.empty(n)
    for i, r in enumerate(records):
        d = r["ann"].get(plan)
        if d is None:
            rt[i], rec[i] = r["bf"]["rt"], r["bf"]["rec"]
            continue
        j = min(d["min_i"] + steps, len(d["hps"]) - 1)
        rt[i], rec[i] = d["rt"][j], d["rec"][j]
    return {"rt": rt, "rec": rec}


def agg(rt, rec, mask=None):
    if mask is not None:
        rt, rec = rt[mask], rec[mask]
    tot_s = rt.sum() / 1000.0
    return {"n": int(len(rt)), "qps": float(len(rt) / tot_s) if tot_s > 0 else np.nan,
            "avg_recall": float(rec.mean()), "tot_s": float(tot_s),
            "p1": float(np.percentile(rec, 1)), "p10": float(np.percentile(rec, 10)),
            "frac_lt_95": float((rec < 0.95).mean()),
            "frac_lt_90": float((rec < 0.90).mean()),
            "frac_lt_80": float((rec < 0.80).mean())}


def add_bf_cost(records, gt, cfg):
    """Attach BF Stage-3 predicted cost to each record (for a fair decision)."""
    # fit a quick pooled BF Stage-3 on ALL data (used only inside decisions;
    # realised latency always uses measured rt). Cheap & not the eval target.
    swr = sweep_rows(gt, [r["pair"] for r in records])
    s = swr[swr["plan"] == "BF"]
    if cfg["s3_per_dataset"]:
        mdl = {}
        for ds, sg in s.groupby("qt"):
            mdl[ds] = LogOLS().fit(stage3_feats(sg["n_pass"].to_numpy(), sg["rho"].to_numpy(),
                                                sg["N"].to_numpy(), sg["hp"].to_numpy(), False),
                                   sg["rt_ms"].to_numpy())
    else:
        mdl = LogOLS().fit(stage3_feats(s["n_pass"].to_numpy(), s["rho"].to_numpy(),
                                        s["N"].to_numpy(), s["hp"].to_numpy(), True),
                           s["rt_ms"].to_numpy())
    for r in records:
        c = _s3_cost(mdl, cfg, np.array([r["n_pass"]]), np.array([r["rho"]]),
                     np.array([r["N"]]), np.array([0.0]), r["qt"])[0]
        r["bf_cost"] = float(c)


# --------------------------------------------------------------------------- #
# optimizer frontier (gamma x safety-steps Pareto) + operating-point selection
# --------------------------------------------------------------------------- #
GAMMAS = [1.0, 1.15, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0,
          30.0, 50.0, 75.0, 100.0]
STEPS = [0, 1, 2, 3, 99]


def optimizer_points(records, lam=0.0, gammas=GAMMAS, steps=STEPS):
    pts = []
    for st in steps:
        for g in gammas:
            e = eval_policy(records, g, lam, st)
            a = agg(e["rt"], e["rec"])
            a.update({"gamma": g, "steps": st, "is_oracle_pct": 100 * e["is_oracle"].mean()})
            pts.append(a)
    return pts


def pick_op(points, target_recall):
    """Cheapest (max qps) point whose avg_recall >= target; else closest below."""
    ok = [p for p in points if p["avg_recall"] >= target_recall - 1e-9]
    if ok:
        return max(ok, key=lambda p: p["qps"])
    return max(points, key=lambda p: p["avg_recall"])


def pick_op_floor(records, floor_p10, lam=0.0):
    """Among (gamma,steps), cheapest (max qps) whose per-query p10 recall >= floor."""
    best = None
    for st in STEPS:
        for g in GAMMAS:
            e = eval_policy(records, g, lam, st)
            p10 = np.percentile(e["rec"], 10)
            if p10 >= floor_p10 - 1e-9:
                q = len(e["rt"]) / (e["rt"].sum() / 1000.0)
                if best is None or q > best["qps"]:
                    best = {"gamma": g, "steps": st, "qps": q, "p10": p10,
                            "avg_recall": float(e["rec"].mean())}
    return best


# --------------------------------------------------------------------------- #
# per-pair STATIC arrays (aligned to records) for matched-recall comparisons
# --------------------------------------------------------------------------- #
def static_pair_arrays(records, gt, plan, hp):
    n = len(records)
    rt = np.empty(n)
    rec = np.empty(n)
    for i, r in enumerate(records):
        pl = gt[r["pair"]]["plans"].get(plan)
        if pl is None:
            rt[i], rec[i] = np.nan, np.nan
            continue
        hps, rc, tt = pl
        j = np.where(hps == hp)[0]
        if len(j) == 0:
            rt[i], rec[i] = np.nan, np.nan
        else:
            rt[i], rec[i] = tt[j[0]], rc[j[0]]
    return rt, rec


def static_best_op(hnsw_pts, ivf_pts, level):
    """(plan, hp, point) of the fixed-hp static policy with avg_recall>=level and
    max qps; None if the static ceiling is below `level`."""
    cand = [p for p in (hnsw_pts + ivf_pts) if p["avg_recall"] >= level - 1e-9]
    if not cand:
        return None
    p = max(cand, key=lambda d: d["qps"])
    return p["plan"], p["hp"], p


def fold_qps_stats(rt, rec, folds, mask_valid=None):
    """per-fold QPS mean/std for error bars."""
    qs = []
    for f in sorted(set(folds)):
        m = folds == f
        if mask_valid is not None:
            m = m & mask_valid
        if m.sum() == 0:
            continue
        qs.append(m.sum() / (np.nansum(rt[m]) / 1000.0))
    return float(np.mean(qs)), float(np.std(qs))


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# "desired recall is the only user knob": map a target avg-recall -> operating
# point (gamma & hp-safety are INTERNAL) -> achieved QPS and speedups.
# --------------------------------------------------------------------------- #
def recall_op_table(records, opt_pts, static_env, hnsw_env, oroute_env, bf_qps, desired_list):
    n = len(records)
    static_ceiling = max(p["avg_recall"] for p in static_env)
    hnsw_ceiling = max(p["avg_recall"] for p in hnsw_env)
    orc_ceiling = max(p["avg_recall"] for p in oroute_env)
    rows = []
    for d in desired_list:
        op = pick_op(opt_pts, d)
        e = eval_policy(records, op["gamma"], 0.0, op["steps"])
        q = n / (e["rt"].sum() / 1000.0)
        ach = float(e["rec"].mean())
        qs = interp_qps(static_env, ach) if ach <= static_ceiling + 1e-9 else None
        qh = interp_qps(hnsw_env, ach) if ach <= hnsw_ceiling + 1e-9 else None
        # oracle-route QPS at the SAME avg recall (honest "how close to oracle").
        qo = interp_qps(oroute_env, ach) if ach <= orc_ceiling + 1e-9 else None
        rows.append({
            "desired_recall": d, "achieved_recall": round(ach, 4),
            "gamma": op["gamma"], "hp_safety_steps": op["steps"], "qps": round(q, 1),
            "speedup_vs_static_HNSW_pre": (round(q / qh, 2) if qh else np.nan),
            "static_HNSW_pre_feasible": qh is not None,
            "speedup_vs_static_best": (round(q / qs, 2) if qs else np.nan),
            "static_best_feasible": qs is not None,
            "speedup_vs_always_BF": round(q / bf_qps, 1),
            "p10_recall": round(float(np.percentile(e["rec"], 10)), 2),
            "frac_below_0.95": round(float((e["rec"] < 0.95).mean()), 3),
            "pct_of_oracle_qps": (round(100 * q / qo, 0) if qo else np.nan),
        })
    return pd.DataFrame(rows)


def measure_overhead(gt, pairs, base, target, reps=5):
    """Amortised per-query PLANNING cost of the optimizer (model inference only):
    Stage-1 feasibility + Stage-2 hp + Stage-3 cost for every candidate plan. This
    is what the optimizer spends per query *before* any vector search; the reported
    QPS counts only measured search time, so this quantifies how negligible the
    decision overhead is."""
    import time
    M = fit_fold(gt, pairs, target, base)
    subs = []
    for plan in ANN_PLANS:
        for ds in ("movies", "reviews"):
            sp = [p for p in pairs if p[0] == ds and plan in gt[p]["plans"]]
            if not sp:
                continue
            Xr = {"sigma": np.array([gt[p]["sigma"] for p in sp]),
                  "rho": np.array([gt[p]["rho"] for p in sp]),
                  "n_pass": np.array([gt[p]["n_pass"] for p in sp])}
            N = np.array([gt[p]["N"] for p in sp])
            hp = np.full(len(sp), 500.0)
            subs.append((plan, ds, feat_s12(Xr, base["use_logn_s12"]),
                         Xr["n_pass"], Xr["rho"], N, hp))
    t0 = time.perf_counter()
    for _ in range(reps):
        for plan, ds, X, npass, rho, N, hp in subs:
            M["feas"][plan].predict(X)
            hpm = M["hpreg"][plan]
            _ = hpm if isinstance(hpm, float) else hpm.predict(X)
            _s3_cost(M["s3"][plan], base, npass, rho, N, hp, ds)
    dt = (time.perf_counter() - t0) / reps
    return dt / len(pairs) * 1e6  # microseconds per query


# --------------------------------------------------------------------------- #
# reporting: stage diagnostics
# --------------------------------------------------------------------------- #
def report_stage_diag(diag, target):
    rows = []
    print(f"\n--- Stage 1/2 diagnostics (target R>={target}) ---")
    print(f"  {'plan':10s} {'feas_acc':>8s} {'feas_auc':>8s} {'hp_hit':>7s} "
          f"{'n_gate':>6s} | {'uni_acc':>7s} {'uni_hit':>7s}")
    for p in ANN_PLANS:
        acc = float(np.nanmean(diag["s1_acc"][p]))
        au = float(np.nanmean(diag["s1_auc"][p]))
        hn, hd = diag["s2_hit"][p]
        hit = hn / hd if hd else np.nan
        ua = float(np.nanmean(diag["s2u_acc"][p]))
        un, ud = diag["s2u_hit"][p]
        uhit = un / ud if ud else np.nan
        print(f"  {p:10s} {acc:8.3f} {au:8.3f} {hit:7.3f} {hd:6d} | {ua:7.3f} {uhit:7.3f}")
        rows.append({"plan": p, "feas_acc": acc, "feas_auc": au, "hp_hit_rate": hit,
                     "n_gate": hd, "unified_feas_acc": ua, "unified_hp_hit_rate": uhit})
    print("\n--- Stage 3 runtime log-R^2 ---")
    print(f"  {'plan':10s} {'@true_hp':>9s} {'@pred_hp':>9s}")
    s3rows = []
    for p in ALL_PLANS:
        yt, yp = diag["s3t"][p]
        r2t = qo.log_r2(np.array(yt), np.array(yp)) if yt else np.nan
        r2p = np.nan
        if p in diag["s3p"] and diag["s3p"][p][0]:
            r2p = qo.log_r2(np.array(diag["s3p"][p][0]), np.array(diag["s3p"][p][1]))
        print(f"  {p:10s} {r2t:9.3f} {r2p:9.3f}")
        s3rows.append({"plan": p, "logR2_true_hp": r2t, "logR2_pred_hp": r2p})
    return rows, s3rows


# --------------------------------------------------------------------------- #
# matched-recall analysis (headline) + decomposition
# --------------------------------------------------------------------------- #
def matched_recall_table(records, gt, opt_pts, hnsw_static, ivf_static,
                         sp_hnsw, sp_ivf, oracle_route, levels):
    """At each average-recall `level`: optimizer/static-best/single-plan-hp/oracle-
    route QPS (interpolated on their frontiers), the optimizer's realised gain over
    static-best, and the achievable gain decomposed into cheaper-hp vs plan-routing."""
    opt_env = pareto_envelope(opt_pts)
    static_env = pareto_envelope(hnsw_static + ivf_static)
    sp_env = pareto_envelope(sp_hnsw + sp_ivf)
    rows = []
    for L in levels:
        q_opt = interp_qps(opt_env, L)
        q_static = interp_qps(static_env, L)
        q_sp = interp_qps(sp_env, L)
        q_route = interp_qps(oracle_route, L)
        row = {"level": L, "opt_qps": q_opt, "static_best_qps": q_static,
               "single_plan_hp_qps": q_sp, "oracle_route_qps": q_route,
               "realised_gain_vs_static": (q_opt / q_static) if (q_opt and q_static) else np.nan,
               "hp_gain": (q_sp / q_static) if (q_sp and q_static) else np.nan,
               "route_gain": (q_route / q_sp) if (q_route and q_sp) else np.nan,
               "oracle_gain_vs_static": (q_route / q_static) if (q_route and q_static) else np.nan}
        rows.append(row)
    return pd.DataFrame(rows), opt_env, static_env, sp_env


# --------------------------------------------------------------------------- #
# decision quality
# --------------------------------------------------------------------------- #
def decision_quality(records, e, target):
    chosen = e["chosen"]
    rec = e["rec"]
    rt = e["rt"]
    oracle_plan = np.array([r["oracle"][0] for r in records], object)
    oracle_cost = np.array([r["oracle"][1] for r in records])
    regret = rt / np.clip(oracle_cost, 1e-9, None)
    miss = rec < target
    right_plan = chosen == oracle_plan
    # confusion matrix
    conf = pd.crosstab(pd.Series(oracle_plan, name="oracle"),
                       pd.Series(chosen, name="chosen")).reindex(
        index=ALL_PLANS, columns=ALL_PLANS, fill_value=0)
    out = {
        "chosen_is_oracle_pct": float(100 * right_plan.mean()),
        "mean_regret": float(regret.mean()), "median_regret": float(np.median(regret)),
        "p90_regret": float(np.percentile(regret, 90)),
        "recall_fail_rate": float(miss.mean()),
        "miss_same_plan_pct": float(100 * (miss & right_plan).mean()),
        "miss_diff_plan_pct": float(100 * (miss & ~right_plan).mean()),
        "confusion": conf, "regret": regret,
    }
    return out


# --------------------------------------------------------------------------- #
# breakdowns
# --------------------------------------------------------------------------- #
def breakdown_table(records, gt, e_opt, static_plan, static_hp, target):
    """Per-dataset and per-selectivity-bin: QPS, avg recall, win-rate vs static-best."""
    st_rt, st_rec = static_pair_arrays(records, gt, static_plan, static_hp)
    qt = np.array([r["qt"] for r in records])
    sig = np.array([r["sigma"] for r in records])
    binid = np.clip(np.digitize(sig, SEL_BINS) - 1, 0, len(SEL_LABELS) - 1)
    opt_rt, opt_rec = e_opt["rt"], e_opt["rec"]
    # win: optimizer faster AND recall not materially worse (>= min(static, target))
    valid = ~np.isnan(st_rt)
    win = valid & (opt_rt < st_rt) & (opt_rec >= np.minimum(st_rec, target) - 1e-9)
    rows = []
    for scope, mask in ([("ALL", np.ones(len(records), bool))]
                        + [(ds, qt == ds) for ds in ("movies", "reviews")]):
        for b in range(len(SEL_LABELS)):
            m = mask & (binid == b)
            if m.sum() == 0:
                continue
            v = m & valid
            rows.append({
                "scope": scope, "sel_bin": SEL_LABELS[b], "n": int(m.sum()),
                "opt_qps": float(m.sum() / (opt_rt[m].sum() / 1000)),
                "static_qps": float(v.sum() / (st_rt[v].sum() / 1000)) if v.sum() else np.nan,
                "opt_avg_recall": float(opt_rec[m].mean()),
                "static_avg_recall": float(st_rec[v].mean()) if v.sum() else np.nan,
                "win_rate_vs_static": float(win[v].mean()) if v.sum() else np.nan,
            })
        m = mask
        v = m & valid
        rows.append({
            "scope": scope, "sel_bin": "ALL", "n": int(m.sum()),
            "opt_qps": float(m.sum() / (opt_rt[m].sum() / 1000)),
            "static_qps": float(v.sum() / (st_rt[v].sum() / 1000)) if v.sum() else np.nan,
            "opt_avg_recall": float(opt_rec[m].mean()),
            "static_avg_recall": float(st_rec[v].mean()) if v.sum() else np.nan,
            "win_rate_vs_static": float(win[v].mean()) if v.sum() else np.nan,
        })
    return pd.DataFrame(rows), win, st_rt, st_rec


# --------------------------------------------------------------------------- #
# ablations
# --------------------------------------------------------------------------- #
def base_cfg(backend="cart", min_leaf=30):
    return {"backend": backend, "min_leaf": min_leaf, "use_logn_s12": False,
            "s3_per_dataset": False, "max_depth_s12": 3, "per_dataset_s12": False}


def run_ablations(gt, pairs, target, level, base):
    """Vary one design choice at a time; report realised optimizer QPS at matched
    avg-recall `level` and recall-fail-rate at default gamma."""
    variants = {
        "base (sigma,rho; gate; pooled+logN; CART)": dict(base),
        "+log(n_pass) in Stage1-2": {**base, "use_logn_s12": True},
        "Stage3 per-dataset (no logN)": {**base, "s3_per_dataset": True},
    }
    if HAVE_SKLEARN:
        variants["sklearn DecisionTree"] = {**base, "backend": "sk-tree"}
        variants["sklearn RandomForest"] = {**base, "backend": "sk-rf"}
    if HAVE_LGBM:
        variants["LightGBM"] = {**base, "backend": "lgbm"}
    if HAVE_XGB:
        variants["XGBoost"] = {**base, "backend": "xgb"}
    rows = []
    for name, cfg in variants.items():
        recs, diag = cv_records(gt, pairs, target, cfg)
        add_bf_cost(recs, gt, cfg)
        pts = optimizer_points(recs)
        env = pareto_envelope(pts)
        q = interp_qps(env, level)
        # default gamma=1.3 steps=0
        e = eval_policy(recs, 1.3, 0.0, 0)
        a = agg(e["rt"], e["rec"])
        s1 = float(np.nanmean([np.nanmean(diag["s1_auc"][p]) for p in ANN_PLANS]))
        # lambda / gate is a policy/label choice; capture unified-hit as gate proxy
        rows.append({"variant": name, f"opt_qps@r{level}": q,
                     "default_qps": a["qps"], "default_avg_recall": a["avg_recall"],
                     "default_recall_fail": a["frac_lt_95"], "mean_feas_auc": s1})
    # lambda sweep + gate(classifier vs unified) + safety-bump on base
    recs, _ = cv_records(gt, pairs, target, base)
    add_bf_cost(recs, gt, base)
    lam_rows = []
    for lam in (0.0, 1.0, 2.0, 4.0, 8.0):
        e = eval_policy(recs, 1.3, lam, 0)
        a = agg(e["rt"], e["rec"])
        lam_rows.append({"lambda": lam, "qps": a["qps"], "avg_recall": a["avg_recall"],
                         "recall_fail": a["frac_lt_95"],
                         "chosen_is_oracle_pct": 100 * e["is_oracle"].mean()})
    bump_rows = []
    for st in (0, 1, 2, 99):
        e = eval_policy(recs, 1.3, 0.0, st)
        a = agg(e["rt"], e["rec"])
        bump_rows.append({"safety_steps": st, "qps": a["qps"],
                          "avg_recall": a["avg_recall"], "recall_fail": a["frac_lt_95"]})
    return pd.DataFrame(rows), pd.DataFrame(lam_rows), pd.DataFrame(bump_rows)


# --------------------------------------------------------------------------- #
# recall-target sensitivity
# --------------------------------------------------------------------------- #
def run_sensitivity(gt, pairs, targets, base):
    rows = []
    for t in targets:
        recs, _ = cv_records(gt, pairs, t, base)
        add_bf_cost(recs, gt, base)
        pts = optimizer_points(recs)
        env = pareto_envelope(pts)
        # static-best & oracle-route at this target
        hn = static_frontier(gt, pairs, "HNSW-pre")
        iv = static_frontier(gt, pairs, "IVF-pre")
        static_env = pareto_envelope(hn + iv)
        route = oracle_route_frontier(gt, pairs, [t])
        static_ceiling = max(p["avg_recall"] for p in static_env)
        # match at the static ceiling (level both can reach) and at target
        for L, tag in [(min(t, static_ceiling), "match@min(t,ceil)"), (t, f"match@{t}")]:
            q_opt = interp_qps(env, L)
            q_static = interp_qps(static_env, L)
            rows.append({"target": t, "match_level": round(L, 3), "tag": tag,
                         "opt_qps": q_opt, "static_best_qps": q_static,
                         "gain_vs_static": (q_opt / q_static) if (q_opt and q_static) else np.nan,
                         "static_ceiling": static_ceiling,
                         "oracle_route_qps": route[0]["qps"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# cost-model honesty: verify + fallback-to-BF, and the online hp-ramp comparison
# --------------------------------------------------------------------------- #
def cost_model_honesty(records, gt, target, op):
    """Upper-bound the true cost of silent recall misses.

    v2 commits to (plan, hp*) and can miss recall silently. A safe deployment would
    VERIFY recall and, on a miss, FALL BACK to exact BF. Upper bound (worst case:
    every miss pays a full extra BF pass):
        corrected_rt = predicted_rt + (BF_rt if realised recall < target else 0)
    This restores recall to >= target on every pair (misses become BF, recall 1.0).

    Compared against:
      * predict-hp-directly (no correction)      -- what v2 reports (risky)
      * always-BF                                -- the safe floor
      * online hp RAMP: climb the grid from the bottom until recall>=target (paying
        every intermediate hp) then stop; on a plan that never reaches target, add
        a BF fallback. This is the v1-style ramp cost (no under-prediction risk)."""
    e = eval_policy(records, op["gamma"], 0.0, op["steps"])
    rt, rec, chosen = e["rt"], e["rec"], e["chosen"]
    miss = rec < target
    bf_rt = np.array([r["bf"]["rt"] for r in records])
    corrected = rt + np.where(miss, bf_rt, 0.0)
    corrected_rec = np.where(miss, 1.0, rec)

    # online ramp for the chosen plan at each pair
    ramp_rt = np.empty(len(records))
    ramp_rec = np.empty(len(records))
    for i, r in enumerate(records):
        pl = chosen[i]
        if pl == "BF":
            ramp_rt[i], ramp_rec[i] = r["bf"]["rt"], r["bf"]["rec"]
            continue
        hps, rc, tt = gt[r["pair"]]["plans"][pl]
        ok = np.where(rc >= target)[0]
        if len(ok):
            j = ok[0]
            ramp_rt[i] = tt[:j + 1].sum()   # pay every hp up to & incl. feasible
            ramp_rec[i] = rc[j]
        else:
            ramp_rt[i] = tt.sum() + r["bf"]["rt"]  # full ramp then BF fallback
            ramp_rec[i] = 1.0

    n = len(records)
    def q(x):
        return n / (x.sum() / 1000.0)
    return {
        "predict_direct": {"qps": q(rt), "avg_recall": float(rec.mean()),
                           "recall_fail": float(miss.mean())},
        "verify_fallback_BF": {"qps": q(corrected), "avg_recall": float(corrected_rec.mean()),
                               "recall_fail": float((corrected_rec < target).mean()),
                               "extra_cost_pct": float(100 * (corrected.sum() - rt.sum()) / rt.sum())},
        "online_ramp": {"qps": q(ramp_rt), "avg_recall": float(ramp_rec.mean()),
                        "recall_fail": float((ramp_rec < target).mean())},
        "always_BF": {"qps": q(bf_rt), "avg_recall": 1.0, "recall_fail": 0.0},
        "op": op,
    }


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_frontier(out_png, gls, opt_env, opt_pure, hnsw_static, ivf_static,
                  sp_hnsw, oroute, bf_pt, oracle_pt, hnsw_pred_pt, mtab, key=None,
                  scope="both datasets pooled"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11, 7.5))

    def xy(pts):
        s = sorted(pts, key=lambda d: d["avg_recall"])
        return [d["avg_recall"] for d in s], [d["qps"] for d in s]

    x, y = xy(hnsw_static)
    ax.plot(x, y, "s-", color=PLAN_PLOT_COLORS["HNSW-pre"], ms=7, lw=1.8,
            label="STATIC HNSW-pre (fixed ef, all queries)")
    x, y = xy(ivf_static)
    ax.plot(x, y, "^-", color=PLAN_PLOT_COLORS["IVF-pre"], ms=7, lw=1.8,
            label="STATIC IVF-pre (fixed nprobe, all queries)")
    x, y = xy(pareto_envelope(hnsw_static + ivf_static))
    ax.plot(x, y, "-", color="#333", lw=3.0, alpha=0.75, label="static-best (envelope)")
    x, y = xy(sp_hnsw)
    ax.plot(x, y, ":", color=PLAN_PLOT_COLORS["HNSW-pre"], lw=2.0,
            label="HNSW-pre + per-query ORACLE hp (hp lever, no routing)")
    x, y = xy(oroute)
    ax.plot(x, y, ":", color=PLAN_PLOT_COLORS["IVF-pre"], lw=2.0,
            label="oracle route (per-query best plan+hp)")
    x, y = xy(opt_pure)
    ax.plot(x, y, "-", color="#9ec5ff", lw=1.4, alpha=0.9, label="optimizer (gamma only)")
    x, y = xy(opt_env)
    ax.plot(x, y, "o-", color=PLAN_PLOT_COLORS["HNSW-pre"], ms=6, lw=2.6,
            markerfacecolor="white", markeredgewidth=1.6,
            label="OPTIMIZER (learned; gamma x hp-safety)")
    ax.scatter([bf_pt[0]], [bf_pt[1]], marker="s", s=150, color=PLAN_PLOT_COLORS["BF"],
               zorder=6, label=f"always-BF (recall 1.0, {bf_pt[1]:.0f} QPS)")
    ax.scatter([oracle_pt[0]], [oracle_pt[1]], marker="D", s=140, color="#39c07a",
               zorder=6, label=f"oracle @R>=0.95 ({oracle_pt[1]:.0f} QPS)")
    ax.scatter([hnsw_pred_pt[0]], [hnsw_pred_pt[1]], marker="X", s=150, color="#b07ad6",
               zorder=6, label="always-HNSW-pre @ OUR predicted hp (old baseline)")
    ax.axvline(0.95, color="#888", ls="--", lw=1, alpha=0.8)
    ax.text(0.9505, ax.get_ylim()[0] * 1.1 if ax.get_ylim()[0] > 0 else 7,
            "R*=0.95", color="#666", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel(f"Average recall over (query,filter) pairs   [{scope}]")
    ax.set_ylabel("Throughput [QPS, log]")
    ax.set_title(f"Throughput vs average recall -- {gls} GLS -- DATASET: {scope.upper()}\n"
                 "static single-plan frontiers plateau at the HNSW feasibility ceiling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.0, loc="lower left", framealpha=0.95)

    # KEY-number annotation: user sets desired recall; gamma is internal.
    # Headline speedup is vs the STATIC HNSW-pre baseline at the same recall (BF is
    # absurdly conservative, so vs-BF is only a footnote).
    if key is not None:
        KC = "#ff7f0e"  # distinct orange (green is used by IVF/oracle)
        d, q = key["desired_recall"], key["qps"]
        ax.axvline(d, color=KC, ls="--", lw=1.4, alpha=0.9)
        ax.scatter([d], [q], marker="*", s=520, color=KC, edgecolors="white",
                   linewidths=1.4, zorder=9)
        hq = key.get("hnsw_qps")
        if hq:
            # speedup guide from static HNSW-pre up to the optimizer at desired recall
            ax.annotate("", xy=(d, q), xytext=(d, hq),
                        arrowprops=dict(arrowstyle="<->", color=KC, lw=1.6, alpha=0.9))
            ax.text(d - 0.004, np.sqrt(q * hq),
                    f"{key['speedup_vs_static_HNSW_pre']:.2f}x\nvs HNSW-pre", color=KC,
                    fontsize=9, fontweight="bold", ha="right", va="center")
        vh = (f"{key['speedup_vs_static_HNSW_pre']:.2f}x vs static HNSW-pre @ same recall"
              if key.get("hnsw_feasible") else "static HNSW-pre INFEASIBLE here (routing to BF needed)")
        vs_static = (f"{key['speedup_vs_static_best']:.2f}x vs static-best"
                     if key.get("static_best_feasible") else "static-best INFEASIBLE here")
        pct = key["pct_of_oracle_qps"]
        pct_s = ("" if pct is None or (isinstance(pct, float) and np.isnan(pct))
                 else f"; {pct:.0f}% of oracle-route @ same recall")
        kbox = (f"KEY RESULT  (user sets desired recall; gamma is internal)\n"
                f"desired avg-recall = {d:g}  ->  optimizer {q:.0f} QPS\n"
                f"   = {vh}\n"
                f"   = {vs_static}{pct_s}\n"
                f"(vs always-BF = {key['speedup_vs_always_BF']:.0f}x, not the headline; "
                f"planning overhead ~{key['overhead_us']:.1f} us/query)")
        ax.text(0.985, 0.985, kbox, transform=ax.transAxes, va="top", ha="right",
                fontsize=9.2, bbox=dict(boxstyle="round", fc="#fff3e6", ec=KC, lw=1.6))

    # matched-recall gains (secondary, honest)
    def _fin(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))
    txt = ["matched-recall gain (optimizer / static-best):"]
    for _, r in mtab.iterrows():
        g, oq, sq = r["realised_gain_vs_static"], r["opt_qps"], r["static_best_qps"]
        if _fin(oq) and _fin(sq):
            txt.append(f"  @avg-recall {r['level']:.3g}: {g:.2f}x  "
                       f"(opt {oq:.0f} / static {sq:.0f} QPS)")
        elif _fin(oq):
            txt.append(f"  @avg-recall {r['level']:.3g}: static INFEASIBLE (ceiling); "
                       f"opt {oq:.0f} QPS")
        else:
            txt.append(f"  @avg-recall {r['level']:.3g}: below optimizer's cheapest op")
    ax.text(0.015, 0.985, "\n".join(txt), transform=ax.transAxes, va="top", ha="left",
            fontsize=8.4, bbox=dict(boxstyle="round", fc="#fff8e6", ec="#e6c76a"))
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved frontier plot -> {out_png}")


def plot_matched_bar(out_png, gls, records, gt, opt_pts, hnsw_static, ivf_static,
                     sp_env, oroute, levels):
    plt = _mpl()
    folds = np.array([r["fold"] for r in records])
    methods = ["static-best", "single-plan +hp (oracle)", "optimizer (learned)",
               "oracle route"]
    colors = ["#555", "#b07ad6", "#5aa0ff", "#39c07a"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(levels))
    w = 0.2
    all_vals = []
    infeasible = []  # (x, method_index)
    for mi, meth in enumerate(methods):
        vals, errs = [], []
        for li, L in enumerate(levels):
            if meth == "static-best":
                sbo = static_best_op(hnsw_static, ivf_static, L)
                if sbo is None:
                    vals.append(np.nan); errs.append(0.0); infeasible.append((li, mi)); continue
                rt, rec = static_pair_arrays(records, gt, sbo[0], sbo[1])
                m, s = fold_qps_stats(rt, rec, folds, ~np.isnan(rt))
                vals.append(m); errs.append(s)
            elif meth == "optimizer (learned)":
                op = pick_op(opt_pts, L)
                e = eval_policy(records, op["gamma"], 0.0, op["steps"])
                if e["rec"].mean() < L - 0.02:
                    vals.append(np.nan); errs.append(0.0); infeasible.append((li, mi)); continue
                m, s = fold_qps_stats(e["rt"], e["rec"], folds)
                vals.append(m); errs.append(s)
            elif meth.startswith("single-plan"):
                q = interp_qps(sp_env, L)
                vals.append(q if q else np.nan); errs.append(0.0)
                if not q:
                    infeasible.append((li, mi))
            else:
                q = interp_qps(oroute, L)
                vals.append(q if q else np.nan); errs.append(0.0)
                if not q:
                    infeasible.append((li, mi))
        all_vals += [v for v in vals if not np.isnan(v)]
        ax.bar(x + (mi - 1.5) * w, vals, w, yerr=errs, capsize=3,
               color=colors[mi], label=meth, alpha=0.92)
    lo = min(all_vals) * 0.75
    hi = max(all_vals) * 1.25
    ax.set_ylim(lo, hi)
    for li, mi in infeasible:
        ax.text(x[li] + (mi - 1.5) * w, lo * 1.05, "infeasible\n(ceiling)", rotation=90,
                ha="center", va="bottom", fontsize=6.5, color="#b00")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"avg recall\n= {L:.3g}" for L in levels])
    ax.set_ylabel("Throughput [QPS, log]  (bars: fold mean +/- s.d.)")
    ax.set_title(f"Matched-recall throughput -- {gls} GLS\n"
                 "static single-plan is INFEASIBLE at avg-recall 0.95 (feasibility ceiling "
                 "~0.94); optimizer & oracle-route are not")
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved matched-recall bar -> {out_png}")


def plot_recall_cdf(out_png, gls, opt_rec, opt_lbl, static_rec, static_lbl,
                    oracle_rec, target):
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for rec, lbl, c in [(opt_rec, opt_lbl, "#5aa0ff"),
                        (static_rec[~np.isnan(static_rec)], static_lbl, "#333"),
                        (oracle_rec, "oracle @R>=0.95", "#39c07a")]:
        s = np.sort(rec)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, cdf, lw=2.2, color=c, label=lbl)
    ax.axvline(target, color="#e2585f", ls="--", lw=1, label=f"R*={target}")
    ax.axvline(0.90, color="#e6a23c", ls=":", lw=1)
    ax.set_xlabel("Per-query realised recall")
    ax.set_ylabel("Cumulative fraction of (query,filter) pairs")
    ax.set_title(f"Realised recall CDF -- {gls} GLS")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    # tail fraction table
    ax2.axis("off")
    rows = [["method", "avg", "p1", "p10", "<0.95", "<0.90", "<0.80"]]
    for rec, lbl in [(opt_rec, opt_lbl), (static_rec[~np.isnan(static_rec)], static_lbl),
                     (oracle_rec, "oracle")]:
        rows.append([lbl.split(" (")[0][:20], f"{rec.mean():.3f}",
                     f"{np.percentile(rec,1):.2f}", f"{np.percentile(rec,10):.2f}",
                     f"{(rec<0.95).mean()*100:.0f}%", f"{(rec<0.90).mean()*100:.0f}%",
                     f"{(rec<0.80).mean()*100:.0f}%"])
    t = ax2.table(cellText=rows, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    for j in range(len(rows[0])):
        t[0, j].set_facecolor("#eef3fb")
        t[0, j].set_text_props(weight="bold")
    ax2.set_title("Recall distribution tails (fraction of pairs below threshold)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved recall CDF -> {out_png}")


def plot_decision_scatter(out_png, gls, records, e, win, st_valid, Lmatch, sb_lbl):
    plt = _mpl()
    sig = np.array([r["sigma"] for r in records])
    rho = np.array([r["rho"] for r in records])
    qt = np.array([r["qt"] for r in records])
    chosen = e["chosen"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    for j, ds in enumerate(["movies", "reviews"]):
        dm = qt == ds
        # row 0: win/loss vs static-best
        ax = axes[0][j]
        for lbl, mask, c in [("optimizer WINS", dm & st_valid & win, "#39c07a"),
                             ("optimizer LOSES", dm & st_valid & ~win, "#e2585f")]:
            ax.scatter(sig[mask], rho[mask], s=20, alpha=0.55, c=c, edgecolors="none",
                       label=f"{lbl} ({int(mask.sum())})")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"win/loss vs static-best [{sb_lbl}] @avg-recall~{Lmatch:.2f} -- {ds}")
        ax.legend(fontsize=8, loc="lower left")
        if j == 0:
            ax.set_ylabel("GLS correlation rho")
        # row 1: chosen plan
        ax = axes[1][j]
        for plan in ALL_PLANS:
            mask = dm & (chosen == plan)
            if mask.sum() == 0:
                continue
            ax.scatter(sig[mask], rho[mask], s=20, alpha=0.55,
                       c=PLAN_PLOT_COLORS[plan], edgecolors="none",
                       label=f"{plan} ({int(mask.sum())})")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"optimizer chosen plan -- {ds}")
        ax.set_xlabel("Filter selectivity (log)")
        ax.legend(fontsize=8, loc="lower left", title="plan (count)")
        if j == 0:
            ax.set_ylabel("GLS correlation rho")
    fig.suptitle(f"Decision map & win/loss vs static-best -- {gls} GLS", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved decision scatter -> {out_png}")


def plot_selectivity_heatmap(out_png, gls, bd, Lmatch):
    plt = _mpl()
    sub = bd[(bd["scope"].isin(["movies", "reviews"])) & (bd["sel_bin"] != "ALL")].copy()
    sub["speedup"] = sub["opt_qps"] / sub["static_qps"]
    piv_sp = sub.pivot(index="sel_bin", columns="scope", values="speedup").reindex(SEL_LABELS)
    piv_win = sub.pivot(index="sel_bin", columns="scope", values="win_rate_vs_static").reindex(SEL_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, piv, title, fmt, cmap, vlim in [
            (axes[0], piv_sp, f"QPS speedup opt/static-best @~{Lmatch:.2f}", "{:.2f}x", "RdYlGn", (0.5, 2.0)),
            (axes[1], piv_win, "win-rate vs static-best", "{:.0%}", "RdYlGn", (0, 1))]:
        data = piv.to_numpy(float)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("dataset")
        ax.set_ylabel("filter selectivity bin")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Optimizer vs static-best by selectivity & dataset -- {gls} GLS", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved selectivity heatmap -> {out_png}")


def _make_key(kr, hnsw_env, key_recall, overhead_us):
    return {"desired_recall": key_recall, "qps": kr["qps"],
            "speedup_vs_static_HNSW_pre": kr["speedup_vs_static_HNSW_pre"],
            "hnsw_feasible": bool(kr["static_HNSW_pre_feasible"]),
            "hnsw_qps": (interp_qps(hnsw_env, key_recall)
                         if kr["static_HNSW_pre_feasible"] else None),
            "speedup_vs_static_best": kr["speedup_vs_static_best"],
            "static_best_feasible": bool(kr["static_best_feasible"]),
            "speedup_vs_always_BF": kr["speedup_vs_always_BF"],
            "pct_of_oracle_qps": kr["pct_of_oracle_qps"], "overhead_us": overhead_us}


SP_TARGETS = [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.0]


def build_dataset_frontier(out, gls, gt, pairs, records, target, key_recall,
                           overhead_us, scope, tag):
    """Compute all frontier ingredients on a (possibly dataset-filtered) subset and
    emit one throughput-vs-recall plot. Returns (key, rop) for that scope. Stage-1/2/3
    predictions in `records` come from the pooled-trained models (as shipped); only the
    evaluation set is restricted to the subset."""
    hnsw_static = static_frontier(gt, pairs, "HNSW-pre")
    ivf_static = static_frontier(gt, pairs, "IVF-pre")
    sp_hnsw = single_plan_hp_frontier(gt, pairs, "HNSW-pre", SP_TARGETS)
    sp_ivf = single_plan_hp_frontier(gt, pairs, "IVF-pre", SP_TARGETS)
    oroute = oracle_route_frontier(gt, pairs, SP_TARGETS)
    static_env = pareto_envelope(hnsw_static + ivf_static)
    static_ceiling = max(p["avg_recall"] for p in static_env)
    hnsw_env = pareto_envelope(hnsw_static)
    oroute_env = pareto_envelope(oroute)
    n = len(records)
    bf_rt = np.array([gt[p]["plans"]["BF"][2][0] for p in pairs])
    bf_qps = n / (bf_rt.sum() / 1000.0)
    oracle_pt_d = oracle_route_frontier(gt, pairs, [target])[0]
    opt_pts = optimizer_points(records)
    opt_pure = optimizer_points(records, steps=[0])
    hnsw_pred_a = agg(*[eval_single_plan(records, "HNSW-pre", 0)[k] for k in ("rt", "rec")])
    levels = sorted({0.91, 0.93, round(static_ceiling, 3), 0.95})
    mtab, opt_env, static_env, sp_env = matched_recall_table(
        records, gt, opt_pts, hnsw_static, ivf_static, sp_hnsw, sp_ivf, oroute, levels)
    desired_grid = sorted({0.90, 0.92, round(static_ceiling, 3), 0.95, key_recall})
    rop = recall_op_table(records, opt_pts, static_env, hnsw_env, oroute_env, bf_qps, desired_grid)
    rop.to_csv(out / f"full_recall_to_op_{tag}.csv", index=False)
    krow = rop[np.isclose(rop["desired_recall"], key_recall)]
    kr = (krow.iloc[0] if len(krow) else recall_op_table(
        records, opt_pts, static_env, hnsw_env, oroute_env, bf_qps, [key_recall]).iloc[0]).to_dict()
    key = _make_key(kr, hnsw_env, key_recall, overhead_us)
    plot_frontier(out / f"full_frontier_{tag}.png", gls, opt_env, opt_pure, hnsw_static,
                  ivf_static, sp_hnsw, oroute, (1.0, bf_qps),
                  (oracle_pt_d["avg_recall"], oracle_pt_d["qps"]),
                  (hnsw_pred_a["avg_recall"], hnsw_pred_a["qps"]), mtab, key=key, scope=scope)
    return key, rop


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_one(path, gls, out, base, target=RT_DEFAULT, sens_targets=(0.90, 0.95, 0.99),
            fast=False, key_recall=0.90):
    print(f"\n{'='*78}\n RIGOROUS BENEFIT ANALYSIS  |  GLS = {gls}  |  {path}\n{'='*78}")
    sweep = load_sweep(path)
    gt = build_gt(sweep)
    pairs = sorted(gt.keys())
    n = len(pairs)
    nmov = sum(p[0] == "movies" for p in pairs)
    print(f"  {n} (query,filter) pairs  (movies={nmov}, reviews={n-nmov})")
    base_rate = {pl: float(np.mean([feas_besthp(*gt[p]["plans"][pl][:2], target)[0]
                                    for p in pairs if pl in gt[p]["plans"]]))
                 for pl in ANN_PLANS}
    print("  feasibility base rate (some hp reaches R>=%.2f): %s" % (
        target, ", ".join(f"{k}={v:.3f}" for k, v in base_rate.items())))

    # ---- ground-truth frontiers (fixed policies / bounds) ----
    hnsw_static = static_frontier(gt, pairs, "HNSW-pre")
    ivf_static = static_frontier(gt, pairs, "IVF-pre")
    sp_targets = [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.0]
    sp_hnsw = single_plan_hp_frontier(gt, pairs, "HNSW-pre", sp_targets)
    sp_ivf = single_plan_hp_frontier(gt, pairs, "IVF-pre", sp_targets)
    oroute = oracle_route_frontier(gt, pairs, sp_targets)
    sp_env = pareto_envelope(sp_hnsw + sp_ivf)
    static_env = pareto_envelope(hnsw_static + ivf_static)
    static_ceiling = max(p["avg_recall"] for p in static_env)
    bf_rt = np.array([gt[p]["plans"]["BF"][2][0] for p in pairs])
    bf_qps = n / (bf_rt.sum() / 1000.0)
    oracle_pt_d = oracle_route_frontier(gt, pairs, [target])[0]
    print(f"  static-best avg-recall CEILING = {static_ceiling:.4f} "
          f"(single fixed-hp policy cannot exceed this)")
    print(f"  always-BF = {bf_qps:.1f} QPS (recall 1.0);  "
          f"oracle @R>=%.2f = {oracle_pt_d['qps']:.1f} QPS" % target)
    pd.DataFrame(hnsw_static + ivf_static).drop(columns=["policy"]).to_csv(
        out / f"full_static_frontier_{gls}.csv", index=False)

    # ---- learned optimizer CV ----
    records, diag = cv_records(gt, pairs, target, base)
    add_bf_cost(records, gt, base)
    folds = np.array([r["fold"] for r in records])
    s12, s3 = report_stage_diag(diag, target)
    pd.DataFrame(s12).to_csv(out / f"full_stage12_{gls}.csv", index=False)
    pd.DataFrame(s3).to_csv(out / f"full_stage3_{gls}.csv", index=False)

    opt_pts = optimizer_points(records)
    opt_pure = optimizer_points(records, steps=[0])
    hnsw_pred = eval_single_plan(records, "HNSW-pre", 0)
    hnsw_pred_a = agg(hnsw_pred["rt"], hnsw_pred["rec"])

    # ---- matched-recall (headline) ----
    # levels must lie in BOTH frontiers' range: optimizer's cheapest op already
    # yields ~0.905 avg recall, static tops out at the ~0.94 feasibility ceiling.
    levels = sorted({0.91, 0.93, round(static_ceiling, 3), 0.95})
    mtab, opt_env, static_env, sp_env = matched_recall_table(
        records, gt, opt_pts, hnsw_static, ivf_static, sp_hnsw, sp_ivf, oroute, levels)
    mtab.to_csv(out / f"full_matched_recall_{gls}.csv", index=False)
    print("\n--- MATCHED-RECALL: optimizer vs static-best (QPS at equal avg recall) ---")
    for _, r in mtab.iterrows():
        sb = "INFEASIBLE" if (r["static_best_qps"] is None or np.isnan(
            r["realised_gain_vs_static"])) else f"{r['static_best_qps']:.0f} QPS"
        g = "n/a" if np.isnan(r["realised_gain_vs_static"]) else f"{r['realised_gain_vs_static']:.2f}x"
        print(f"  @avg-recall {r['level']:.2f}: optimizer {r['opt_qps'] or float('nan'):.0f} QPS  "
              f"vs static-best {sb}  -> gain {g}   "
              f"[decompose: hp {r['hp_gain']:.2f}x x route {r['route_gain']:.2f}x = "
              f"{r['oracle_gain_vs_static']:.2f}x achievable]")

    # matched per-query FLOOR p10>=0.90
    floor_op = pick_op_floor(records, 0.90)
    static_floor_ok = any(p["p10"] >= 0.90 for p in hnsw_static + ivf_static)
    print(f"\n--- MATCHED per-query FLOOR (p10 recall >= 0.90) ---")
    if floor_op:
        print(f"  optimizer reaches p10>=0.90 at {floor_op['qps']:.1f} QPS "
              f"(gamma={floor_op['gamma']:g}, steps={floor_op['steps']}, "
              f"avg recall {floor_op['avg_recall']:.3f})")
    print(f"  static single-plan reaches p10>=0.90? {static_floor_ok}  "
          f"(HNSW/IVF p10 ceiling = {max(p['p10'] for p in hnsw_static+ivf_static):.2f})")
    print(f"  always-BF p10=1.0 at {bf_qps:.1f} QPS")

    # ---- operating points for downstream ----
    #  * Lmatch: matched AVERAGE recall (mid-band) for the win/loss comparison.
    #  * op_hi : honest high-recall op = cheapest with per-query FLOOR p10>=0.90
    #           (avoids the deceptive "avg>=0.95 but 23% per-query misses" point).
    #  * op_cheap: the cheap default (gamma=1.3) where hp-errors dominate.
    Lmatch = round(min(static_ceiling, 0.93), 3)
    opM = pick_op(opt_pts, Lmatch)
    eM = eval_policy(records, opM["gamma"], 0.0, opM["steps"])
    sbo = static_best_op(hnsw_static, ivf_static, Lmatch)
    op_hi = floor_op if floor_op else pick_op(opt_pts, 0.95)
    e_hi = eval_policy(records, op_hi["gamma"], 0.0, op_hi["steps"])
    e_cheap = eval_policy(records, 1.3, 0.0, 0)

    # ---- decision quality (per-query-safe op AND cheap op) ----
    dq = decision_quality(records, e_hi, target)
    dqc = decision_quality(records, e_cheap, target)
    print(f"\n--- DECISION QUALITY ---")
    print(f"  [safe op: per-query p10>=0.90; gamma={op_hi['gamma']:g}, steps={op_hi['steps']}, "
          f"avg recall {e_hi['rec'].mean():.3f}]")
    print(f"    chosen==oracle {dq['chosen_is_oracle_pct']:.1f}% ; regret mean "
          f"{dq['mean_regret']:.2f}/med {dq['median_regret']:.2f}/p90 {dq['p90_regret']:.2f} ; "
          f"recall-fail {100*dq['recall_fail_rate']:.1f}% "
          f"(hp-error {dq['miss_same_plan_pct']:.1f}% / wrong-plan {dq['miss_diff_plan_pct']:.1f}%)")
    print(f"  [cheap op: gamma=1.3, steps=0, avg recall {e_cheap['rec'].mean():.3f}]")
    print(f"    chosen==oracle {dqc['chosen_is_oracle_pct']:.1f}% ; regret mean "
          f"{dqc['mean_regret']:.2f} ; recall-fail {100*dqc['recall_fail_rate']:.1f}% "
          f"(hp-error {dqc['miss_same_plan_pct']:.1f}% / wrong-plan {dqc['miss_diff_plan_pct']:.1f}%)")
    dq["confusion"].to_csv(out / f"full_confusion_{gls}.csv")

    # ---- breakdowns (per dataset & selectivity) at Lmatch ----
    bd, win, st_rt, st_rec = breakdown_table(records, gt, eM, sbo[0], sbo[1], target)
    bd.to_csv(out / f"full_breakdown_{gls}.csv", index=False)
    print(f"\n--- BREAKDOWN win-rate vs static-best [{sbo[0]}@{sbo[1]:g}] "
          f"@avg-recall~{Lmatch} (opt avg recall {eM['rec'].mean():.3f}) ---")
    for _, r in bd[bd["sel_bin"] == "ALL"].iterrows():
        print(f"  {r['scope']:8s} n={r['n']:4d}  opt {r['opt_qps']:6.1f} QPS  "
              f"static {r['static_qps']:6.1f} QPS  win-rate {100*r['win_rate_vs_static']:.0f}%")

    # ---- ablations (incl. CART vs sklearn-tree/RF vs LightGBM vs XGBoost) ----
    if fast:
        ab = _read_csv(out / f"full_ablations_{gls}.csv")
        lam_df = _read_csv(out / f"full_lambda_sweep_{gls}.csv")
        bump_df = _read_csv(out / f"full_bump_sweep_{gls}.csv")
        sens = _read_csv(out / f"full_sensitivity_{gls}.csv")
        print("\n(--fast: loaded cached ablations + sensitivity CSVs)")
    else:
        ab, lam_df, bump_df = run_ablations(gt, pairs, target, Lmatch, base)
        ab.to_csv(out / f"full_ablations_{gls}.csv", index=False)
        lam_df.to_csv(out / f"full_lambda_sweep_{gls}.csv", index=False)
        bump_df.to_csv(out / f"full_bump_sweep_{gls}.csv", index=False)
        print(f"\n--- ABLATIONS (optimizer QPS @ matched avg-recall {Lmatch}) ---")
        print(ab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        # ---- sensitivity to recall target ----
        sens = run_sensitivity(gt, pairs, list(sens_targets), base)
        sens.to_csv(out / f"full_sensitivity_{gls}.csv", index=False)
        print(f"\n--- SENSITIVITY to recall target ---")
        print(sens.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- cost-model honesty ----
    cm = cost_model_honesty(records, gt, target, {"gamma": op_hi["gamma"], "steps": op_hi["steps"]})
    cm_rows = [{"policy": k, **v} for k, v in cm.items() if isinstance(v, dict)]
    pd.DataFrame(cm_rows).to_csv(out / f"full_costmodel_{gls}.csv", index=False)
    print(f"\n--- COST-MODEL HONESTY (silent misses; per-query-safe op) ---")
    for k in ("predict_direct", "verify_fallback_BF", "online_ramp", "always_BF"):
        v = cm[k]
        extra = f"  (+{v.get('extra_cost_pct', 0):.0f}% cost)" if "extra_cost_pct" in v else ""
        print(f"  {k:20s}: {v['qps']:7.1f} QPS  avg recall {v['avg_recall']:.3f}  "
              f"fail {100*v['recall_fail']:.1f}%{extra}")

    # ---- desired-recall -> operating point (gamma is INTERNAL) + KEY speedup ----
    overhead_us = measure_overhead(gt, pairs, base, target)
    oroute_env = pareto_envelope(oroute)
    hnsw_env = pareto_envelope(hnsw_static)
    desired_grid = sorted({0.90, 0.92, round(static_ceiling, 3), 0.95, key_recall})
    rop = recall_op_table(records, opt_pts, static_env, hnsw_env, oroute_env, bf_qps, desired_grid)
    rop.to_csv(out / f"full_recall_to_op_{gls}.csv", index=False)
    krow = rop[np.isclose(rop["desired_recall"], key_recall)]
    if krow.empty:
        krow = recall_op_table(records, opt_pts, static_env, hnsw_env, oroute_env,
                               bf_qps, [key_recall])
    kr = krow.iloc[0].to_dict()
    key = _make_key(kr, hnsw_env, key_recall, overhead_us)
    print(f"\n--- KEY: desired recall is the only user knob (gamma is internal) ---")
    print(f"  optimizer planning overhead ~{overhead_us:.2f} us/query "
          f"(searches are ms-scale -> QPS is unaffected)")
    print(rop.to_string(index=False))
    _vh = (f"{kr['speedup_vs_static_HNSW_pre']:.2f}x vs static HNSW-pre"
           if kr["static_HNSW_pre_feasible"] else "static HNSW-pre infeasible at this recall")
    _pct = kr["pct_of_oracle_qps"]
    _pct_s = "" if _pct is None or np.isnan(_pct) else f", {_pct:.0f}% of oracle-route @ same recall"
    print(f"  >>> KEY @ desired avg-recall {key_recall:g}: {kr['qps']:.0f} QPS = "
          f"{_vh}{_pct_s}  (vs always-BF {kr['speedup_vs_always_BF']:.0f}x, not the headline)")

    # ---- plots ----
    plot_frontier(out / f"full_frontier_{gls}.png", gls, opt_env, opt_pure, hnsw_static,
                  ivf_static, sp_hnsw, oroute, (1.0, bf_qps),
                  (oracle_pt_d["avg_recall"], oracle_pt_d["qps"]),
                  (hnsw_pred_a["avg_recall"], hnsw_pred_a["qps"]), mtab, key=key,
                  scope="both datasets pooled")
    # ---- per-dataset frontiers (so it is unambiguous which dataset) ----
    ds_keys = {}
    for ds in ("movies", "reviews"):
        pd_pairs = [p for p in pairs if p[0] == ds]
        pd_recs = [r for r in records if r["qt"] == ds]
        if pd_pairs and pd_recs:
            k_ds, _ = build_dataset_frontier(out, gls, gt, pd_pairs, pd_recs, target,
                                             key_recall, overhead_us, f"{ds} only",
                                             f"{gls}_{ds}")
            ds_keys[ds] = k_ds
    plot_matched_bar(out / f"full_matched_bar_{gls}.png", gls, records, gt, opt_pts,
                     hnsw_static, ivf_static, sp_env, oroute, levels)
    oracle_rec = np.array([r["oracle"][2] for r in records])
    sb_lbl = f"{sbo[0]}@{sbo[1]:g}"
    plot_recall_cdf(out / f"full_recall_cdf_{gls}.png", gls, eM["rec"],
                    f"optimizer (avg {eM['rec'].mean():.3f})", st_rec,
                    f"static-best {sb_lbl} (avg {np.nanmean(st_rec):.3f})", oracle_rec, target)
    plot_decision_scatter(out / f"full_decision_scatter_{gls}.png", gls, records, eM,
                          win, ~np.isnan(st_rt), Lmatch, sb_lbl)
    plot_selectivity_heatmap(out / f"full_selectivity_{gls}.png", gls, bd, Lmatch)

    return {
        "gls": gls, "n": n, "bf_qps": bf_qps, "oracle_qps": oracle_pt_d["qps"],
        "static_ceiling": static_ceiling, "hnsw_pred": hnsw_pred_a,
        "matched": mtab, "floor_op": floor_op, "static_floor_ok": static_floor_ok,
        "static_p10_ceiling": float(max(p["p10"] for p in hnsw_static + ivf_static)),
        "op_hi": op_hi, "e_hi_recall": float(e_hi["rec"].mean()),
        "e_hi_qps": float(n / (e_hi["rt"].sum() / 1000.0)), "dq": dq, "dq_cheap": dqc,
        "breakdown": bd, "ablations": ab, "sensitivity": sens, "costmodel": cm,
        "base_rate": base_rate, "Lmatch": Lmatch, "sbo": sb_lbl, "eM_recall": float(eM["rec"].mean()),
        "recall_op": rop, "key": key, "key_recall": key_recall, "overhead_us": overhead_us,
        "ds_keys": ds_keys,
    }


def write_findings(summaries, out):
    md = ["# FANNS query optimizer -- rigorous benefit analysis (v2)", ""]
    md.append("Honest, matched-recall quantification of when/how much the per-query "
              "optimizer beats simple STATIC single-plan policies. Grouped 5-fold CV "
              "(split by query); numpy-CART models (parity with the shipped optimizer). "
              "Run on exact and estimated GLS.")
    md.append("")
    for s in summaries:
        g = s["gls"]
        md.append(f"## {g} GLS")
        md.append("")
        md.append(f"- **{s['n']} (query,filter) pairs.** Feasibility base rate "
                  f"(some hp reaches R>=0.95): " +
                  ", ".join(f"{k} {v:.2f}" for k, v in s["base_rate"].items()) + ".")
        md.append(f"- **Static single-plan hits an avg-recall ceiling of "
                  f"{s['static_ceiling']:.3f}** (HNSW-pre @ ef=1000): one fixed hp for "
                  f"every query cannot go higher, because ~29% of pairs are infeasible "
                  f"for HNSW at any ef. Per-query p10 recall ceiling for static is "
                  f"{s['static_p10_ceiling']:.2f}.")
        k = s["key"]
        _vh = (f"{k['speedup_vs_static_HNSW_pre']:.2f}x vs static HNSW-pre @ same recall"
               if k["hnsw_feasible"] else "static HNSW-pre infeasible at this recall (routing to BF needed)")
        _vs = (f"{k['speedup_vs_static_best']:.2f}x vs static-best"
               if k["static_best_feasible"] else "static-best is infeasible at this recall")
        _pct = k["pct_of_oracle_qps"]
        _pct_s = ("" if _pct is None or (isinstance(_pct, float) and np.isnan(_pct))
                  else f", {_pct:.0f}% of oracle-route throughput at the same recall")
        md.append(f"- **KEY (desired recall is the only user knob; gamma is internal):** "
                  f"at a user-desired **avg recall {k['desired_recall']:g}** the optimizer runs "
                  f"**{k['qps']:.0f} QPS = {_vh}** ({_vs}{_pct_s}). "
                  f"The {k['speedup_vs_always_BF']:.0f}x-vs-always-BF figure is *not* the headline "
                  f"(BF is absurdly conservative). Planning overhead ~{s['overhead_us']:.1f} us/query "
                  f"(ms-scale searches), so QPS is unaffected by the optimizer's own cost.")
        md.append("- **Matched-recall gain of the optimizer over static-best:**")
        for _, r in s["matched"].iterrows():
            if r["static_best_qps"] is None or np.isnan(r["realised_gain_vs_static"]):
                md.append(f"    - @avg-recall {r['level']:.2f}: static-best is "
                          f"**infeasible** (above the {s['static_ceiling']:.3f} ceiling); "
                          f"the optimizer reaches it at {r['opt_qps']:.0f} QPS "
                          f"(vs always-BF {s['bf_qps']:.0f}, oracle {s['oracle_qps']:.0f}).")
            else:
                md.append(f"    - @avg-recall {r['level']:.2f}: "
                          f"**{r['realised_gain_vs_static']:.2f}x** "
                          f"(optimizer {r['opt_qps']:.0f} vs static-best "
                          f"{r['static_best_qps']:.0f} QPS); achievable upper bound "
                          f"{r['oracle_gain_vs_static']:.2f}x = hp {r['hp_gain']:.2f}x "
                          f"x routing {r['route_gain']:.2f}x.")
        fo = s["floor_op"]
        md.append(f"- **Matched per-query floor (p10 recall >= 0.90):** static "
                  f"single-plan **cannot** reach it (p10 ceiling {s['static_p10_ceiling']:.2f}); "
                  + (f"the optimizer reaches it at {fo['qps']:.0f} QPS "
                     f"(avg recall {fo['avg_recall']:.3f}) vs always-BF {s['bf_qps']:.0f} QPS."
                     if fo else "the optimizer also cannot reach it in-grid."))
        cm = s["costmodel"]
        md.append(f"- **Cost-model honesty:** at R>=0.95 the direct predict-hp policy "
                  f"runs {cm['predict_direct']['qps']:.0f} QPS but silently misses recall on "
                  f"{100*cm['predict_direct']['recall_fail']:.0f}% of pairs; a verify+"
                  f"fallback-to-BF correction restores recall and still runs "
                  f"{cm['verify_fallback_BF']['qps']:.0f} QPS "
                  f"(+{cm['verify_fallback_BF']['extra_cost_pct']:.0f}% cost), vs an online "
                  f"hp-ramp at {cm['online_ramp']['qps']:.0f} QPS and always-BF "
                  f"{cm['always_BF']['qps']:.0f} QPS.")
        md.append(f"- **Decision quality** (per-query-safe op, p10>=0.90): chosen==oracle "
                  f"{s['dq']['chosen_is_oracle_pct']:.0f}%, mean regret "
                  f"{s['dq']['mean_regret']:.2f}; at this op the misses are dominated by "
                  f"routing/feasibility errors ({s['dq']['miss_diff_plan_pct']:.0f}% wrong-plan) "
                  f"rather than hp under-prediction, whereas at the cheap op (gamma=1.3) hp "
                  f"under-prediction dominates ({s['dq_cheap']['miss_same_plan_pct']:.0f}% "
                  f"same-plan hp-error).")
        if isinstance(s.get("ablations"), pd.DataFrame) and not s["ablations"].empty:
            col = [c for c in s["ablations"].columns if c.startswith("opt_qps@")][0]
            ab = s["ablations"].dropna(subset=[col])
            # compare ONLY the model-backend variants (not the feature/Stage3 configs)
            bk_names = ("base", "sklearn DecisionTree", "sklearn RandomForest",
                        "LightGBM", "XGBoost")
            abk = ab[ab["variant"].str.startswith(bk_names)]
            if len(abk):
                best = abk.loc[abk[col].idxmax()]
                base_row = abk[abk["variant"].str.startswith("base")]
                base_q = float(base_row[col].iloc[0]) if len(base_row) else np.nan
                others = ", ".join(f"{r['variant'].replace(' (sigma,rho; gate; pooled+logN; CART)','')}"
                                   f" {r[col]:.0f}" for _, r in abk.iterrows())
                md.append(f"- **Model backend** (CART vs sklearn-tree/RF vs LightGBM vs "
                          f"XGBoost), optimizer QPS @ matched recall: CART base {base_q:.0f}, "
                          f"best = **{best['variant'].replace(' (sigma,rho; gate; pooled+logN; CART)','')}** "
                          f"({best[col]:.0f} QPS). The tree-ensembles do **not** beat CART here "
                          f"(all {others} QPS); with only 2 features (sigma, rho) and a tiny "
                          f"ordinal hp grid there is little for a GBM to exploit, so the headline "
                          f"keeps CART (parity with the shipped optimizer).")
        md.append("")
    md.append("## Honest conclusion")
    md.append("")
    s = summaries[0]
    md.append(f"The optimizer's benefit is **regime-dependent, and the 40x-vs-always-BF "
              f"headline is not the right comparison** (BF is absurdly conservative at "
              f"{s['bf_qps']:.0f} QPS / recall 1.0).")
    md.append("")
    md.append("- **Mid recall (~0.90-0.93, achievable by static single-plan):** the gain "
              "over static-best is *small* -- typically ~1.1-1.4x -- and near the HNSW "
              "feasibility ceiling (~0.94) a static HNSW-pre @ ef=1000 can even *beat* the "
              "learned optimizer, because raising avg recall via the gamma knob substitutes "
              "expensive BF instead of just cranking ef.")
    md.append("- **High recall (avg >= 0.95, or a per-query floor p10 >= 0.90): this is the "
              "regime where the optimizer genuinely matters.** No single static ANN policy "
              "can get there (feasibility ceiling); reaching it *requires* routing the hard "
              "pairs to exact BF, and the optimizer does that far more cheaply than always-BF "
              "and approaches the oracle.")
    md.append("- **Where the gain comes from:** most of the achievable benefit is the "
              "**cheaper per-query hp** lever (feasible pairs need far less ef/nprobe than the "
              "worst-case fixed hp), not HNSW-vs-IVF plan routing; routing mainly adds the "
              "BF escape hatch needed for the high-recall regime.")
    md.append("- **Caveat on the p10 floor vs oracle:** the optimizer reaching a per-query "
              f"floor (p10>=0.90) at ~{s['floor_op']['qps']:.0f} QPS while the oracle is only "
              f"{s['oracle_qps']:.0f} QPS is NOT the optimizer beating the oracle -- p10>=0.90 "
              "is a strictly weaker guarantee than the oracle's per-query R>=0.95. For the SAME "
              "per-query R>=0.95 guarantee the optimizer must route essentially all infeasible "
              "pairs to BF and approaches (never beats) the oracle.")
    md.append("- **Recall quantization (k=10):** per-query recall is a multiple of 0.1, so the "
              "R>=0.95 target is effectively 'perfect 10/10 recall', and targets 0.95 and 0.99 "
              "are identical; the sensitivity sweep therefore only shows two distinct regimes "
              "(0.90 vs 0.95==0.99). This also explains the ~0.71 HNSW/IVF feasibility ceiling.")
    md.append("- **Estimated ~= exact GLS**, so the cheap rho estimator suffices.")
    md.append("")
    (out / "FINDINGS.md").write_text("\n".join(md))
    print(f"\n  wrote findings -> {out / 'FINDINGS.md'}")


HTML_CSS = """
  :root{--bg:#0f1320;--card:#171c2e;--ink:#e7ecf5;--muted:#9aa6c0;--line:#27304a;
    --accent:#5aa0ff;--good:#39c07a;--warn:#e6a23c;--bad:#e2585f;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1000px;margin:0 auto;padding:32px 22px 80px}
  h1{font-size:26px;margin:0 0 4px}
  h2{font-size:18px;margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line);color:#fff}
  h3{font-size:15px;margin:18px 0 6px;color:#cdd6ea}
  .sub{color:var(--muted);margin:0 0 18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}
  p{margin:8px 0} ul{margin:8px 0;padding-left:20px} li{margin:5px 0}
  code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13.5px}
  th,td{padding:7px 9px;text-align:left;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
  td.num,th.num{text-align:right;font-family:ui-monospace,monospace}
  .good{color:var(--good)} .warn{color:var(--warn)} .bad{color:var(--bad)}
  .note{color:var(--muted);font-size:13px}
  img{width:100%;border:1px solid var(--line);border-radius:10px;margin:8px 0;background:#fff}
  .key{background:linear-gradient(135deg,#12321f,#0f2740);border:1.5px solid var(--good);
    border-radius:14px;padding:20px 22px;margin:16px 0}
  .key .big{font-size:40px;font-weight:800;color:var(--good);line-height:1.05}
  .key .lab{color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.05em}
  .kgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:6px}
  .kgrid .n{font-size:22px;font-weight:700;color:#fff}
  .tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;font-weight:600;
    background:rgba(90,160,255,.18);color:#9ec5ff}
  @media(max-width:680px){.kgrid{grid-template-columns:1fr 1fr}}
"""


def _fmt_cell(v):
    if v is None:
        return "&mdash;"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, float):
        if np.isnan(v):
            return "&mdash;"
        if abs(v) >= 10:
            return f"{v:,.1f}"
        return f"{v:.2f}" if abs(v) >= 1 else f"{v:.3f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    return str(v)


def _df_to_html(df, cols=None, headers=None, numeric=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='note'>(no data)</p>"
    cols = cols or list(df.columns)
    headers = headers or cols
    numeric = numeric or set()
    th = "".join(f"<th class='num'>{h}</th>" if c in numeric else f"<th>{h}</th>"
                 for c, h in zip(cols, headers))
    body = []
    for _, r in df.iterrows():
        tds = "".join(
            f"<td class='num'>{_fmt_cell(r[c])}</td>" if c in numeric
            else f"<td>{_fmt_cell(r[c])}</td>" for c in cols)
        body.append(f"<tr>{tds}</tr>")
    return f"<table><tr>{th}</tr>{''.join(body)}</table>"


def write_findings_html(summaries, out, primary="exact"):
    prim = next((s for s in summaries if s["gls"] == primary), summaries[0])
    g = prim["gls"]
    k = prim["key"]
    cm = prim["costmodel"]
    dq, dqc = prim["dq"], prim["dq_cheap"]
    _pct = k["pct_of_oracle_qps"]
    pct_ok = not (_pct is None or (isinstance(_pct, float) and np.isnan(_pct)))
    vh_big = (f"{k['speedup_vs_static_HNSW_pre']:.2f}&times;" if k["hnsw_feasible"] else "routing&gt;BF")

    # KEY banner (headline speedup vs the static HNSW-pre baseline at matched recall)
    banner = f"""
  <div class="key">
    <div class="lab">KEY RESULT &mdash; the user sets a desired recall; &gamma; is chosen internally</div>
    <p style="margin:6px 0 2px">At a user-desired <b>average recall of {k['desired_recall']:g}</b>,
      the optimizer runs at <b>{k['qps']:.0f} QPS</b> (both datasets pooled):</p>
    <div class="kgrid">
      <div><div class="lab">vs static HNSW-pre @ same recall</div><div class="big">{vh_big}</div></div>
      <div><div class="lab">vs static-best</div><div class="n">{('%.2f&times;'%k['speedup_vs_static_best']) if k['static_best_feasible'] else 'n/a'}</div></div>
      <div><div class="lab">of oracle-route @ same recall</div><div class="n">{('%.0f%%'%_pct) if pct_ok else 'n/a'}</div></div>
      <div><div class="lab">planning overhead</div><div class="n">~{k['overhead_us']:.1f} &micro;s/q</div></div>
    </div>
    <p class="note" style="margin-top:10px">The demanding comparison is vs the <b>static HNSW-pre</b>
      single-plan baseline at the same recall &mdash; not always-BF (the {k['speedup_vs_always_BF']:.0f}&times;-vs-BF
      figure is easy: BF is absurdly conservative at {prim['bf_qps']:.0f} QPS / recall 1.0). The oracle
      (per-query cheapest plan+hp) at the same average recall is the honest ceiling. Optimizer planning
      cost (~{k['overhead_us']:.1f}&micro;s: three depth-limited trees + a linear runtime model per candidate
      plan) is negligible against ms-scale searches, so &gamma; can be tuned freely to hit the desired recall.</p>
  </div>
"""

    # recall -> operating point table
    rop = prim["recall_op"]
    rop_html = _df_to_html(
        rop,
        cols=["desired_recall", "achieved_recall", "gamma", "hp_safety_steps", "qps",
              "speedup_vs_static_HNSW_pre", "speedup_vs_static_best", "speedup_vs_always_BF",
              "pct_of_oracle_qps", "p10_recall", "frac_below_0.95"],
        headers=["desired R", "achieved R", "&gamma; (internal)", "hp-safety steps",
                 "QPS", "&times; vs HNSW-pre", "&times; vs static-best", "&times; vs BF",
                 "% of oracle-route", "p10 recall", "frac R&lt;0.95"],
        numeric={"desired_recall", "achieved_recall", "gamma", "hp_safety_steps", "qps",
                 "speedup_vs_static_HNSW_pre", "speedup_vs_static_best", "speedup_vs_always_BF",
                 "pct_of_oracle_qps", "p10_recall", "frac_below_0.95"})

    # matched-recall table
    m = prim["matched"].copy()
    def _gx(v):
        return "&mdash;" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}&times;"
    mrows = []
    for _, r in m.iterrows():
        infeasible = (r["static_best_qps"] is None) or (
            isinstance(r["realised_gain_vs_static"], float) and np.isnan(r["realised_gain_vs_static"]))
        mrows.append({
            "level": r["level"], "opt_qps": r["opt_qps"],
            "static_best_qps": ("infeasible" if infeasible else r["static_best_qps"]),
            "gain": _gx(r["realised_gain_vs_static"]), "hp_gain": _gx(r["hp_gain"]),
            "route_gain": _gx(r["route_gain"]), "oracle_gain": _gx(r["oracle_gain_vs_static"])})
    mdf = pd.DataFrame(mrows)
    matched_html = _df_to_html(
        mdf, cols=["level", "opt_qps", "static_best_qps", "gain", "hp_gain", "route_gain", "oracle_gain"],
        headers=["avg recall", "optimizer QPS", "static-best QPS", "realised gain",
                 "hp lever", "routing lever", "oracle-achievable"],
        numeric={"level", "opt_qps", "static_best_qps"})

    # ablations (backends only for the headline table)
    ab_html = "<p class='note'>(ablations not loaded)</p>"
    if isinstance(prim.get("ablations"), pd.DataFrame) and not prim["ablations"].empty:
        ab = prim["ablations"].copy()
        col = [c for c in ab.columns if c.startswith("opt_qps@")][0]
        ab["variant"] = ab["variant"].str.replace(" (sigma,rho; gate; pooled+logN; CART)", "",
                                                   regex=False)
        ab_html = _df_to_html(ab, cols=["variant", col],
                              headers=["variant / backend", "optimizer QPS @ matched recall"],
                              numeric={col})

    sens = prim.get("sensitivity", pd.DataFrame())
    if isinstance(sens, pd.DataFrame) and not sens.empty:
        _sc = [c for c in ["target", "match_level", "tag", "opt_qps", "static_best_qps",
                           "gain_vs_static", "static_ceiling", "oracle_route_qps"]
               if c in sens.columns]
        _sh = {"target": "recall target", "match_level": "match level", "tag": "matched at",
               "opt_qps": "optimizer QPS", "static_best_qps": "static-best QPS",
               "gain_vs_static": "gain vs static", "static_ceiling": "static ceiling",
               "oracle_route_qps": "oracle-route QPS"}
        sens_html = _df_to_html(sens, cols=_sc, headers=[_sh[c] for c in _sc],
                                numeric=set(_sc) - {"tag"})
    else:
        sens_html = "<p class='note'>(sensitivity not loaded)</p>"

    # cost-model table
    cm_rows = []
    for kk, lab in [("predict_direct", "predict-hp directly (silent misses)"),
                    ("verify_fallback_BF", "verify + fallback-to-BF (recall restored)"),
                    ("online_ramp", "online hp-ramp"),
                    ("always_BF", "always-BF")]:
        v = cm[kk]
        cm_rows.append({"policy": lab, "qps": v["qps"], "avg_recall": v["avg_recall"],
                        "fail": 100 * v["recall_fail"],
                        "extra": (f"+{v['extra_cost_pct']:.0f}%" if "extra_cost_pct" in v else "&mdash;")})
    cm_html = _df_to_html(
        pd.DataFrame(cm_rows), cols=["policy", "qps", "avg_recall", "fail", "extra"],
        headers=["policy", "QPS", "avg recall", "% recall-fail", "extra cost vs direct"],
        numeric={"qps", "avg_recall", "fail"})

    # per-dataset frontier blocks
    def _ds_caption(dk):
        vh = (f"{dk['speedup_vs_static_HNSW_pre']:.2f}&times; vs static HNSW-pre"
              if dk["hnsw_feasible"] else "static HNSW-pre infeasible (routing to BF needed)")
        return (f"at desired recall {dk['desired_recall']:g}: {dk['qps']:.0f} QPS "
                f"= {vh}, {dk['speedup_vs_always_BF']:.0f}&times; vs always-BF.")
    ds_blocks = ""
    for ds in ("movies", "reviews"):
        dk = prim.get("ds_keys", {}).get(ds)
        if dk:
            ds_blocks += (f'<h3>{ds} only</h3>\n'
                          f'<img src="full_frontier_{g}_{ds}.png" alt="frontier {ds} ({g} GLS)">\n'
                          f'<p class="note">{_ds_caption(dk)}</p>\n')

    est_line = ""
    if len(summaries) == 2:
        e = next(s for s in summaries if s["gls"] == "exact")
        se = next(s for s in summaries if s["gls"] == "estimated")
        est_line = (f"<p class='note'><b>Estimated &asymp; exact GLS.</b> Static ceiling "
                    f"{e['static_ceiling']:.3f} (exact) vs {se['static_ceiling']:.3f} (estimated); "
                    f"key speedup vs HNSW-pre {e['key']['speedup_vs_static_HNSW_pre']:.2f}&times; vs "
                    f"{se['key']['speedup_vs_static_HNSW_pre']:.2f}&times; &mdash; the cheap &rho; "
                    f"estimator suffices.</p>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FANNS query optimizer &mdash; benefit analysis</title>
<style>{HTML_CSS}</style></head><body><div class="wrap">

  <h1>FANNS query optimizer &mdash; rigorous benefit analysis
    <span class="note">(v2, FAISS)</span></h1>
  <p class="sub">Honest, matched-recall quantification of when and how much the per-query optimizer
    beats simple static single-plan policies. Grouped 5-fold CV (split by query); numpy-CART models
    (parity with the shipped optimizer). Headline figures on <b>{g} GLS</b>; estimated GLS confirms.</p>

  {banner}

  <h2>Desired recall is the only user knob</h2>
  <div class="card">
    <p>The user picks a target average recall; the optimizer maps it to an internal robustness margin
      <span class="mono">&gamma;</span> (and hp-safety steps) that hits it. Because the reported QPS
      already reflects only measured search time and the planning overhead is ~{k['overhead_us']:.1f}&micro;s/query,
      the mapping is essentially free.</p>
    <h3>Both datasets pooled</h3>
    <img src="full_frontier_{g}.png" alt="throughput vs recall frontier ({g} GLS, pooled)">
    <p class="note">The optimizer curve (blue) tracks the oracle-routing bound (grey). The orange star and
      arrow mark the KEY operating point at the desired recall and its speedup over the static HNSW-pre
      baseline at the same recall (always-BF, red, is shown only for reference).</p>
    {ds_blocks}
    <h3>Desired recall &rarr; operating point &rarr; achieved speedup <span class="note">(pooled)</span></h3>
    {rop_html}
  </div>

  <h2>Matched-recall gain over static-best</h2>
  <div class="card">
    <p>At <b>equal average recall</b>, optimizer QPS vs the upper envelope of the HNSW-pre and IVF-pre
      fixed-hp frontiers. The achievable gain decomposes into a <b>cheaper-hp</b> lever and a
      <b>plan-routing</b> lever.</p>
    {matched_html}
    <img src="full_matched_bar_{g}.png" alt="matched-recall QPS bars ({g} GLS)">
    <p class="note">Static single-plan hits an avg-recall ceiling of {prim['static_ceiling']:.3f}
      (HNSW-pre @ ef=1000): ~29% of pairs are infeasible for HNSW at any ef, so above the ceiling the
      static baseline is simply unavailable and only routing-to-BF (optimizer / oracle) can reach the target.</p>
  </div>

  <h2>Realised per-query recall distribution</h2>
  <div class="card">
    <img src="full_recall_cdf_{g}.png" alt="recall CDF ({g} GLS)">
    <p class="note">Average recall hides per-query misses; the optimizer is compared to static-best and the
      oracle at a matched point. A per-query floor (p10 recall &ge; 0.90) is the honest high-recall target,
      which static single-plan cannot reach (p10 ceiling {prim['static_p10_ceiling']:.2f}).</p>
  </div>

  <h2>Decision quality</h2>
  <div class="card">
    <ul>
      <li><b>Per-query-safe op</b> (p10&ge;0.90, avg recall {prim['e_hi_recall']:.3f}):
        chosen==oracle <b>{dq['chosen_is_oracle_pct']:.0f}%</b>, mean regret {dq['mean_regret']:.2f};
        misses dominated by routing/feasibility errors ({dq['miss_diff_plan_pct']:.0f}% wrong-plan).</li>
      <li><b>Cheap op</b> (&gamma;=1.3, steps=0):
        chosen==oracle {dqc['chosen_is_oracle_pct']:.0f}%, recall-fail {100*dqc['recall_fail_rate']:.0f}%,
        dominated by hp under-prediction ({dqc['miss_same_plan_pct']:.0f}% same-plan hp-error).</li>
    </ul>
    <img src="full_decision_scatter_{g}.png" alt="decision scatter ({g} GLS)">
  </div>

  <h2>Per-selectivity / per-dataset breakdown</h2>
  <div class="card">
    <img src="full_selectivity_{g}.png" alt="selectivity heatmap ({g} GLS)">
  </div>

  <h2>Model backend ablation <span class="note">(CART vs sklearn / LightGBM / XGBoost)</span></h2>
  <div class="card">
    {ab_html}
    <p class="note">Tree-ensembles (LightGBM, XGBoost, sklearn RF) do <b>not</b> beat the numpy CART here:
      with only 2 features (&sigma;, &rho;) and a tiny ordinal hp grid there is little for a GBM to exploit,
      so the headline keeps CART for parity with the shipped optimizer.</p>
  </div>

  <h2>Recall-target sensitivity</h2>
  <div class="card">
    {sens_html}
    <p class="note">Recall is quantized (k=10 &rArr; multiples of 0.1), so the R&ge;0.95 target is effectively
      "perfect 10/10" and targets 0.95 and 0.99 coincide; only 0.90 vs 0.95 are distinct regimes.</p>
  </div>

  <h2>Cost-model honesty</h2>
  <div class="card">
    <p>Predicting hp directly can miss recall silently. An upper bound on the true cost adds a
      verify + fallback-to-BF correction:</p>
    {cm_html}
    <p class="note">Even after paying for verification and BF fallback on the missed pairs, predicting hp
      directly stays well above the online hp-ramp and always-BF.</p>
  </div>

  <h2>Honest conclusion</h2>
  <div class="card">
    <ul>
      <li>The <b>{k['speedup_vs_always_BF']:.0f}&times;-vs-always-BF headline is real but easy</b> &mdash;
        BF is absurdly conservative ({prim['bf_qps']:.0f} QPS / recall 1.0). The demanding comparison is
        vs the static <b>HNSW-pre</b> single-plan baseline at the same recall.</li>
      <li><b>Mid recall (~0.90&ndash;0.93):</b> gain over static HNSW-pre / static-best is <b>small</b>
        (~1.1&ndash;1.4&times;); near the ceiling a static HNSW-pre @ ef=1000 can even beat the optimizer.</li>
      <li><b>High recall (avg &ge; 0.95, or p10 &ge; 0.90):</b> the regime where the optimizer genuinely
        matters &mdash; no single static ANN policy can get there; it requires routing hard pairs to BF, which
        the optimizer does far more cheaply than always-BF and close to the oracle.</li>
      <li><b>Where the gain comes from:</b> mostly the cheaper per-query hp lever, not HNSW-vs-IVF routing;
        routing mainly supplies the BF escape hatch for the high-recall regime.</li>
    </ul>
    {est_line}
  </div>

  <p class="note">Generated by <span class="mono">analysis/qo_analysis_full.py</span> &mdash; plots and CSVs
    in this folder. Reproduce: <span class="mono">python qo_analysis_full.py --gls both</span>.</p>

</div></body></html>"""
    (out / "FINDINGS.html").write_text(html)
    print(f"  wrote findings -> {out / 'FINDINGS.html'}")


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results",
                    default=str(root / "plots/query_optimizer/all_query_results.csv"))
    ap.add_argument("--gls", choices=["exact", "estimated", "both"], default="both")
    ap.add_argument("--model", choices=["cart", "sk-tree", "sk-rf", "lgbm", "xgb"],
                    default="cart",
                    help="model backend for the HEADLINE optimizer (default CART = "
                         "parity with the shipped qo_prototype.py; CART vs sklearn-tree/RF "
                         "vs LightGBM vs XGBoost are all compared in the ablation)")
    ap.add_argument("--min-leaf", type=int, default=30)
    ap.add_argument("--target", type=float, default=RT_DEFAULT)
    ap.add_argument("--fast", action="store_true",
                    help="reuse cached ablation/sensitivity CSVs (fast plot/findings regen)")
    ap.add_argument("--key-recall", type=float, default=0.90,
                    help="desired avg recall for the headline KEY speedup number")
    ap.add_argument("--out", default=str(root / "plots/qo_prototype"))
    args = ap.parse_args()

    _avail = {"cart": True, "sk-tree": HAVE_SKLEARN, "sk-rf": HAVE_SKLEARN,
              "lgbm": HAVE_LGBM, "xgb": HAVE_XGB}
    if not _avail.get(args.model, False):
        raise SystemExit(f"backend '{args.model}' not installed; available: {BACKENDS}")
    print(f"model backends available: {BACKENDS}  (headline = {args.model})")

    exact_path = Path(args.results)
    est_path = exact_path.parent / "gls_est" / "all_query_results.csv"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = base_cfg(args.model, args.min_leaf)

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

    summaries = [run_one(p, lbl, out, base, args.target, fast=args.fast,
                         key_recall=args.key_recall)
                 for lbl, p in targets]
    write_findings(summaries, out)
    write_findings_html(summaries, out)

    if len(summaries) == 2:
        e, s = summaries
        print(f"\n{'='*78}\n exact vs estimated GLS\n{'='*78}")
        print(f"  static avg-recall ceiling : exact {e['static_ceiling']:.3f}  "
              f"estimated {s['static_ceiling']:.3f}")
        me = e["matched"]; ms = s["matched"]
        common = sorted(set(me["level"]).intersection(set(ms["level"])))
        for L in common:
            ge = me[me["level"] == L]["realised_gain_vs_static"]
            gs = ms[ms["level"] == L]["realised_gain_vs_static"]
            if len(ge) and len(gs) and not (np.isnan(ge.iloc[0]) or np.isnan(gs.iloc[0])):
                print(f"  gain vs static @{L:.3g}  : exact {ge.iloc[0]:.2f}x  "
                      f"estimated {gs.iloc[0]:.2f}x")
    print(f"\nOutputs -> {out}")


if __name__ == "__main__":
    main()
