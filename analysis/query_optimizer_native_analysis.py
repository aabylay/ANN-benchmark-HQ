#!/usr/bin/env python3
"""Query-optimizer analysis for the CaseLaw_7M native workload.

Consumes results under::

    results/CaseLaw_7M_native_cases/10/{algo}/...

Joins per-query filter / selectivity / split from sample_manifest.json,
filtered GT from cases_native.hdf5, exact GLS (L=100) from filter_stats_0_k100.csv,
and tiered GLS estimates from gls_est_native_cases.csv.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ann_benchmarks.corpora import caselaw_recall_at_k, pack_root, result_dir
from query_optimizer_analysis import (
    FAISS_ALL_ALGOS,
    RECALL_TARGET,
    _accuracy_metrics,
    analyze_hyperparam_recommendations,
    compute_system_oracle_speedup,
    df_is_k10_only,
    find_best_hyperparams,
    generate_scatter_plots,
    load_gls_estimator_pairs,
    plot_faiss_all_qps_recall_frontier,
    plot_gls_estimator_comparison,
    plot_qps_recall_frontier,
    plot_system_speedup,
    recall_tag,
    verify_brute_force,
)
from query_optimizer_hard_analysis import (
    _algo_slug,
    _attach_tertile,
    plot_tertile_qps_recall_per_algorithm,
)


def parse_native_hdf5_path(file: Path) -> Tuple[Optional[str], Optional[dict]]:
    """Parse .../CaseLaw_7M_native_cases/{k}/{algo}/cases_*.hdf5."""
    try:
        algo = file.parent.name
        k = int(file.parent.parent.name)
    except (IndexError, ValueError):
        return None, None

    patterns = [
        ("hnsw", r"cases_(?:euclidean|angular)_M_(\d+)_efConstruction_(\d+)_(\d+)\.hdf5"),
        ("hnsw", r"cases_(?:euclidean|angular)_(\d+)_M_(\d+)_efConstruction_(\d+)_(\d+)\.hdf5"),
        ("ivf", r"cases_(?:euclidean|angular)_clusters_(\d+)_(\d+)\.hdf5"),
        ("ivf", r"cases_(?:euclidean|angular)_(\d+)_nlist_(\d+)_(\d+)\.hdf5"),
        ("bf", r"cases_(?:euclidean|angular)(?:_\d+)?\.hdf5"),
        ("bf", r"cases_(?:euclidean|angular)_0\.hdf5"),
    ]
    for idx_type, pattern in patterns:
        match = re.match(pattern, file.name)
        if not match:
            continue
        params: dict = {
            "query_type": "cases",
            "algo": algo,
            "k": k,
        }
        if idx_type == "hnsw":
            if match.lastindex == 3:
                m_val, ef_c, ef_s = match.groups()
            else:
                _, m_val, ef_c, ef_s = match.groups()
            params.update(
                {
                    "m": int(m_val),
                    "ef_construction": int(ef_c),
                    "ef_search": int(ef_s),
                    "hyperparam": int(ef_s),
                    "hyperparam_name": "ef_search",
                }
            )
        elif idx_type == "ivf":
            if match.lastindex == 2:
                clusters, probes = match.groups()
            else:
                _, clusters, probes = match.groups()
            params.update(
                {
                    "clusters": int(clusters),
                    "probes": int(probes),
                    "hyperparam": int(probes),
                    "hyperparam_name": "probes",
                }
            )
        else:
            params.update({"hyperparam": 0, "hyperparam_name": "none"})
        return idx_type, params
    return None, None


def _vectorized_recalls(
    neighbors: np.ndarray,
    gt_k: np.ndarray,
    denom: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Fast vectorized recall@k matching caselaw_recall_at_k."""
    pred = neighbors[:, :k]
    # Broadcast equality: (n_q, k_pred, 1) == (n_q, 1, k_gt)
    eq = (pred[:, :, None] == gt_k[:, None, :]) & (pred[:, :, None] != -1)
    # Since returned neighbor IDs per query are unique, any(axis=2) is membership in G
    hits = eq.any(axis=2).sum(axis=1).astype(np.float64)
    return hits / denom


def _build_native_results_df(
    results_dir: Path,
    pack_path: Path,
    manifest: dict,
    gls_stats_path: Path,
    gls_est_path: Optional[Path],
    smoke: bool = False,
) -> Tuple[pd.DataFrame, Optional[int]]:
    records = list(manifest["queries"])
    n_population = int(manifest["n_population"])

    with h5py.File(pack_path, "r") as pf:
        all_gt_ids = np.asarray(pf["gt_ids"][:], dtype=np.int32)
        all_nres = np.asarray(pf["nres"][:], dtype=np.int32)

    exact_map: Dict[int, float] = {}
    if gls_stats_path.is_file():
        exact_df = pd.read_csv(gls_stats_path)
        col = "correlation_NEW" if "correlation_NEW" in exact_df.columns else "gls_correlation"
        for qidx, val in zip(exact_df["query_index"], exact_df[col]):
            exact_map[int(qidx)] = float(val)

    est_map: Dict[int, Dict[str, float]] = {}
    if gls_est_path is not None and gls_est_path.is_file():
        est_df = pd.read_csv(gls_est_path)
        q_col = "query_index" if "query_index" in est_df.columns else "q_id"
        q_raw_arr = est_df[q_col].tolist()
        est_c = (
            est_df["gls_correlation_est"].to_numpy(dtype=np.float64)
            if "gls_correlation_est" in est_df.columns
            else est_df["gls_correlation"].to_numpy(dtype=np.float64)
        )
        c1 = est_df["gls_correlation_1"].to_numpy(dtype=np.float64) if "gls_correlation_1" in est_df.columns else np.full(len(est_df), np.nan)
        c10 = est_df["gls_correlation_10"].to_numpy(dtype=np.float64) if "gls_correlation_10" in est_df.columns else np.full(len(est_df), np.nan)
        c100 = est_df["gls_correlation_100"].to_numpy(dtype=np.float64) if "gls_correlation_100" in est_df.columns else np.full(len(est_df), np.nan)
        for idx_r, q_raw in enumerate(q_raw_arr):
            qi = int(str(q_raw).replace("qc", "")) if isinstance(q_raw, str) else int(q_raw)
            est_map[qi] = {
                "gls_correlation_est": float(est_c[idx_r]),
                "gls_correlation_1": float(c1[idx_r]),
                "gls_correlation_10": float(c10[idx_r]),
                "gls_correlation_100": float(c100[idx_r]),
            }

    detected_max_queries: Optional[int] = None
    frames: List[pd.DataFrame] = []
    cached_meta: Dict[Tuple[int, int], dict] = {}

    for file in sorted(results_dir.rglob("*.hdf5")):
        idx_type, params = parse_native_hdf5_path(file)
        if params is None:
            continue
        with h5py.File(file, "r") as f:
            file_max_q = f.attrs.get("max_queries", None)
            if file_max_q is not None:
                file_max_q = int(file_max_q)
            if not smoke and file_max_q is not None:
                detected_max_queries = file_max_q
                continue
            if smoke and file_max_q is not None:
                detected_max_queries = file_max_q
            neighbors = np.asarray(f["neighbors"][:], dtype=np.int32)
            times = np.asarray(f["times"][:], dtype=np.float64)
            if "dist_comps" in f:
                dist_comps = np.asarray(f["dist_comps"][:], dtype=np.float64)
            else:
                dist_comps = np.full(len(times), np.nan, dtype=np.float64)

        expected_n = detected_max_queries if (smoke and detected_max_queries is not None) else n_population
        if len(neighbors) != expected_n:
            if smoke and detected_max_queries is None and len(neighbors) < n_population:
                detected_max_queries = len(neighbors)
                expected_n = len(neighbors)
            else:
                raise RuntimeError(
                    f"{file}: row count {len(neighbors)} != expected {expected_n}"
                )

        k = int(params["k"])
        mkey = (expected_n, k)
        if mkey not in cached_meta:
            sub_records = records[:expected_n]
            qis = np.asarray([int(r["query_index"]) for r in sub_records], dtype=np.int64)
            qids = [f"qc{int(qi):05d}" for qi in qis]
            sels = np.asarray([float(r["selectivity"]) for r in sub_records], dtype=np.float64)
            splits = [str(r["split"]) for r in sub_records]
            nres_sub = np.asarray([int(r.get("nres", all_nres[int(qi)])) for r, qi in zip(sub_records, qis)], dtype=np.int32)
            denom = np.minimum(k, nres_sub).astype(np.float64)
            gt_k = np.full((expected_n, k), -1, dtype=np.int32)
            for pos, (qi, nr) in enumerate(zip(qis, nres_sub)):
                take = min(int(k), int(nr))
                gt_k[pos, :take] = all_gt_ids[int(qi), :take]
            gls_ex = np.asarray([float(exact_map.get(int(qi), np.nan)) for qi in qis], dtype=np.float64)
            gls_est = np.asarray([float(est_map.get(int(qi), {}).get("gls_correlation_est", np.nan)) for qi in qis], dtype=np.float64)
            gls_1 = np.asarray([float(est_map.get(int(qi), {}).get("gls_correlation_1", np.nan)) for qi in qis], dtype=np.float64)
            gls_10 = np.asarray([float(est_map.get(int(qi), {}).get("gls_correlation_10", np.nan)) for qi in qis], dtype=np.float64)
            gls_100 = np.asarray([float(est_map.get(int(qi), {}).get("gls_correlation_100", np.nan)) for qi in qis], dtype=np.float64)
            cached_meta[mkey] = {
                "qis": qis,
                "qids": qids,
                "sels": sels,
                "splits": splits,
                "denom": denom,
                "gt_k": gt_k,
                "gls_ex": gls_ex,
                "gls_est": gls_est,
                "gls_1": gls_1,
                "gls_10": gls_10,
                "gls_100": gls_100,
            }
        meta = cached_meta[mkey]
        recalls = _vectorized_recalls(neighbors, meta["gt_k"], meta["denom"], k=k)
        qps_arr = np.where(times > 0.0, 1.0 / times, np.nan)
        data_dict = {
            "query_id": meta["qids"],
            "query_id_num": meta["qis"],
            "query_type": "cases",
            "filter_id": meta["qis"],
            "filter_selectivity": meta["sels"],
            "k": k,
            "algorithm": params["algo"],
            "recall": recalls,
            "runtime": times,
            "qps": qps_arr,
            "dist_comps": dist_comps[:expected_n],
            "hyperparam": int(params["hyperparam"]),
            "hyperparam_name": params["hyperparam_name"],
            "gls_correlation": meta["gls_ex"],
            "gls_correlation_est": meta["gls_est"],
            "gls_correlation_1": meta["gls_1"],
            "gls_correlation_10": meta["gls_10"],
            "gls_correlation_100": meta["gls_100"],
            "split": meta["splits"],
        }
        if idx_type == "hnsw":
            data_dict["m"] = int(params["m"])
            data_dict["ef_construction"] = int(params["ef_construction"])
            data_dict["ef_search"] = int(params["ef_search"])
        elif idx_type == "ivf":
            data_dict["clusters"] = int(params["clusters"])
            data_dict["probes"] = int(params["probes"])
        frames.append(pd.DataFrame(data_dict))

    if not frames:
        return pd.DataFrame(), detected_max_queries
    return pd.concat(frames, ignore_index=True), detected_max_queries


def _write_gls_summaries(
    df_all: pd.DataFrame,
    exact_stats_path: Path,
    est_stats_path: Path,
    output_dir: Path,
) -> None:
    """Write gls_summary.csv and gls_latency_summary.csv for export_caselaw_numbers.py."""
    pairs = load_gls_estimator_pairs(exact_stats_path, est_stats_path)
    acc = pairs.dropna(subset=["rho_hat", "rho_exact"]).copy()
    if acc.empty:
        raise RuntimeError("No paired GLS rows found for CaseLaw GLS summary")

    est_raw = pd.read_csv(est_stats_path)
    tier_specs = [
        ("gls_correlation_1", "n_b_1", "12-bit (B=4096)"),
        ("gls_correlation_10", "n_b_10", "8-bit (B=256)"),
        ("gls_correlation_100", "n_b_100", "4-bit (B=16)"),
    ]
    summary_rows = []
    rmse0 = float(np.sqrt(np.mean(acc["rho_exact"] ** 2)))
    for col, nb_col, label in tier_specs:
        if col not in acc.columns:
            continue
        sub = acc.dropna(subset=[col, "rho_exact"]).copy()
        sub["rho_hat"] = sub[col]
        m = _accuracy_metrics(sub)
        med_nb = float(est_raw[nb_col].median()) if nb_col in est_raw.columns else float(est_raw["n_b"].median())
        summary_rows.append(
            {
                "workload": "caselaw",
                "tier": label,
                "n_pairs": int(m["n"]),
                "median_n_b": med_nb,
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "bias": float(m["bias"]),
                "pearson": float(m["pearson"]),
                "spearman": float(m["spearman"]),
                "rmse0": rmse0,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "gls_summary.csv", index=False)

    # Latency summary on population estimator latencies, compared against test-split search latency
    df_test = df_all[df_all["split"] == "test"] if "split" in df_all.columns else df_all
    lat_us = pd.to_numeric(est_raw["latency_us"], errors="coerce").dropna()
    lat_us = lat_us[lat_us > 0]
    p50_us = float(lat_us.median())
    mean_us = float(lat_us.mean())
    p90_us = float(np.percentile(lat_us, 90))
    p99_us = float(np.percentile(lat_us, 99))

    bf_rows = df_test[df_test["algorithm"] == "faiss-flat"]
    exact_scan_ms = float(bf_rows["runtime"].mean() * 1000.0) if not bf_rows.empty else np.nan

    ann_rows = df_test[df_test["algorithm"].isin([a for a in FAISS_ALL_ALGOS if a != "faiss-flat"])]
    tau = 0.90
    best_ann_ms = np.nan
    best_algo = ""
    best_hp = 0
    best_succ = np.nan
    if not ann_rows.empty:
        grouped = (
            ann_rows.groupby(["algorithm", "hyperparam"], as_index=False)
            .agg(
                mean_recall=("recall", "mean"),
                success_090=("recall", lambda s: float((s >= tau).mean())),
                mean_runtime_ms=("runtime", lambda s: float(s.mean() * 1000.0)),
            )
        )
        feas = grouped[grouped["mean_recall"] >= tau]
        pick = (
            feas.sort_values("mean_runtime_ms", ascending=True).iloc[0]
            if not feas.empty
            else grouped.sort_values(["mean_recall", "mean_runtime_ms"], ascending=[False, True]).iloc[0]
        )
        best_ann_ms = float(pick["mean_runtime_ms"])
        best_algo = str(pick["algorithm"])
        best_hp = int(pick["hyperparam"])
        best_succ = float(pick["success_090"])

    lat_summary = pd.DataFrame(
        [
            {
                "workload": "caselaw",
                "n": int(len(est_raw)),
                "p50_us": p50_us,
                "mean_us": mean_us,
                "p90_us": p90_us,
                "p99_us": p99_us,
                "best_fixed_ann_ms": best_ann_ms,
                "exact_scan_ms": exact_scan_ms,
                "best_fixed_algorithm": best_algo,
                "best_fixed_hyperparam": best_hp,
                "best_fixed_success_090": best_succ,
                "tau": tau,
                "speedup_vs_ann": (best_ann_ms * 1000.0) / p50_us if p50_us > 0 else np.nan,
                "speedup_vs_bf": (exact_scan_ms * 1000.0) / p50_us if p50_us > 0 else np.nan,
            }
        ]
    )
    lat_summary.to_csv(output_dir / "gls_latency_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=result_dir("caselaw", workload="native", data_table="cases"),
    )
    parser.add_argument(
        "--pack",
        default=str(Path(pack_root("caselaw")) / "native" / "cases_native.hdf5"),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(pack_root("caselaw")) / "sample_manifest.json"),
    )
    parser.add_argument(
        "--gls-stats",
        default=str(Path(pack_root("caselaw")) / "stats" / "filter_stats_0_k100.csv"),
    )
    parser.add_argument(
        "--gls-est",
        default=str(Path(pack_root("caselaw")) / "stats" / "gls_est_native_cases.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/plots/query_optimizer_native_cases",
    )
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    parser.add_argument("--rebuild-csv", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validate truncated smoke results and write to query_optimizer_native_cases_smoke/",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    pack_path = Path(args.pack)
    if not pack_path.is_absolute():
        pack_path = ROOT / pack_path
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    gls_stats_path = Path(args.gls_stats)
    if not gls_stats_path.is_absolute():
        gls_stats_path = ROOT / gls_stats_path
    gls_est_path = Path(args.gls_est)
    if not gls_est_path.is_absolute():
        gls_est_path = ROOT / gls_est_path

    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_population = int(manifest["n_population"])

    if args.smoke:
        output_dir = ROOT / "analysis" / "plots" / "query_optimizer_native_cases_smoke"
        output_dir.mkdir(parents=True, exist_ok=True)
        df, max_q = _build_native_results_df(
            results_dir,
            pack_path,
            manifest,
            gls_stats_path,
            gls_est_path if gls_est_path.is_file() else None,
            smoke=True,
        )
        if df.empty:
            raise SystemExit(f"No smoke results found in {results_dir}")
        expected_n = max_q if max_q is not None else n_population
        distinct_q = df["query_id_num"].nunique()
        if distinct_q != expected_n:
            raise SystemExit(
                f"Smoke distinct query_id_num={distinct_q} != expected={expected_n}"
            )
        expected_prefix = [int(r["query_index"]) for r in manifest["queries"][:expected_n]]
        for algo, g in df.groupby("algorithm"):
            for hp, hg in g.groupby("hyperparam"):
                if hg["query_id_num"].tolist() != expected_prefix:
                    raise SystemExit(
                        f"Row order mismatch for {algo} hp={hp} against manifest prefix"
                    )
        if not df["recall"].between(0.0, 1.0).all():
            raise SystemExit("Recall values outside [0, 1]")
        bf_summary = verify_brute_force(df)
        if bf_summary.empty or not bool(bf_summary["recall_ok"].all()):
            raise SystemExit(f"Brute-force recall check failed:\n{bf_summary}")
        df.to_csv(output_dir / "all_query_results.csv", index=False)
        bf_summary.to_csv(output_dir / "brute_force_recall_check.csv", index=False)
        print(f"Smoke analysis passed ({distinct_q} queries) -> {output_dir}")
        print(bf_summary.to_string(index=False))
        return

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag_dir = output_dir / recall_tag(args.recall_target)
    tag_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "all_query_results.csv"
    reused_csv = False
    if csv_path.is_file() and not args.rebuild_csv:
        print(f"Checking existing {csv_path}")
        df = pd.read_csv(csv_path)
        if int(df["query_id_num"].nunique()) == n_population:
            reused_csv = True
            print(f"Reusing {csv_path}")
        else:
            print(
                f"Existing CSV has {df['query_id_num'].nunique()} distinct queries != "
                f"n_population={n_population}; rebuilding from HDF5 results..."
            )
    if not reused_csv:
        df, max_q = _build_native_results_df(
            results_dir,
            pack_path,
            manifest,
            gls_stats_path,
            gls_est_path if gls_est_path.is_file() else None,
            smoke=False,
        )
        if df.empty:
            if max_q is not None:
                raise SystemExit(
                    f"Refusing paper analysis on truncated smoke results (max_queries={max_q}); "
                    "use --smoke or run full campaign."
                )
            raise SystemExit(f"No native results found under {results_dir}")
        distinct_q = int(df["query_id_num"].nunique())
        if distinct_q != n_population:
            raise SystemExit(
                f"Distinct query_id_num={distinct_q} != manifest n_population={n_population}"
            )
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} ({len(df):,} rows across {distinct_q:,} queries)")

    distinct_q = int(df["query_id_num"].nunique())
    if distinct_q != n_population:
        raise SystemExit(
            f"CSV distinct query_id_num={distinct_q} != manifest n_population={n_population}"
        )

    has_est = "gls_correlation_est" in df.columns and df["gls_correlation_est"].notna().any()
    est_csv = output_dir / "gls_est" / "all_query_results.csv"
    if has_est and (args.rebuild_csv or not reused_csv or not est_csv.is_file()):
        est_csv.parent.mkdir(parents=True, exist_ok=True)
        df_est = df.copy()
        df_est["gls_correlation"] = df_est["gls_correlation_est"]
        df_est.to_csv(est_csv, index=False)
        print(f"Saved {est_csv} ({len(df_est):,} rows)")

    bf_path = output_dir / "brute_force_recall_check.csv"
    if args.rebuild_csv or not reused_csv or not bf_path.is_file():
        bf_summary = verify_brute_force(df)
        bf_summary.to_csv(bf_path, index=False)
        print(bf_summary.to_string(index=False))

    # Section 15: Oracle, runtime, tertile, and optimal-plan figures under
    # query_optimizer_native_cases/ are computed on split == test.
    df_test = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    if df_test.empty:
        raise SystemExit("No test-split rows in all_query_results.csv")

    k10 = df_is_k10_only(df_test)
    best_hp = find_best_hyperparams(df_test, args.recall_target)
    best_hp.to_csv(tag_dir / "best_hyperparameters.csv", index=False)

    hp_sweep, hp_recs = analyze_hyperparam_recommendations(df_test, args.recall_target)
    sweep_path = output_dir / "hyperparam_sweep.csv"
    if args.rebuild_csv or not reused_csv or not sweep_path.is_file():
        hp_sweep.to_csv(sweep_path, index=False)
    hp_recs.to_csv(tag_dir / "hyperparam_recommendations.csv", index=False)

    speedup_df = compute_system_oracle_speedup(df_test, best_hp, args.recall_target)
    speedup_df.to_csv(tag_dir / "system_oracle_speedup.csv", index=False)
    plot_system_speedup(
        speedup_df, tag_dir / "system_oracle_speedup.png", args.recall_target, k10_footnote=k10
    )

    generate_scatter_plots(
        df_test, best_hp, tag_dir, args.recall_target, gls_source="exact", k10_footnote=k10
    )
    plot_qps_recall_frontier(df_test, best_hp, tag_dir, args.recall_target, k10_footnote=k10)
    plot_faiss_all_qps_recall_frontier(
        df_test, best_hp, tag_dir, args.recall_target, k10_footnote=k10
    )

    tertile_dir = output_dir / "qps_recall_tertiles_per_algo"
    tertile_present = tertile_dir.is_dir() and any(tertile_dir.glob("*.png"))
    if args.rebuild_csv or not tertile_present:
        _tagged_exact, edges_gls_exact = _attach_tertile(df_test, "gls_correlation")
        if edges_gls_exact:
            pd.DataFrame([edges_gls_exact]).to_csv(
                output_dir / "tertile_edges_gls_exact.csv", index=False
            )
        per_algo_edges = plot_tertile_qps_recall_per_algorithm(df_test, output_dir)
        if "gls_correlation_est" in per_algo_edges:
            pd.DataFrame([per_algo_edges["gls_correlation_est"]]).to_csv(
                output_dir / "tertile_edges_gls_est.csv", index=False
            )

    if gls_stats_path.is_file() and gls_est_path.is_file():
        eval_needed = (
            args.rebuild_csv
            or not (output_dir / "gls_estimator_accuracy.pdf").is_file()
            or not (output_dir / "gls_summary.csv").is_file()
        )
        if eval_needed:
            plot_gls_estimator_comparison(
                gls_stats_path,
                gls_est_path,
                output_dir,
                runtime_baseline_path=None,
                reference="k=100",
            )
            _write_gls_summaries(df, gls_stats_path, gls_est_path, output_dir)


if __name__ == "__main__":
    main()
