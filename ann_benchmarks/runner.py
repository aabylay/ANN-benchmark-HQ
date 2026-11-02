import argparse
import json
import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple, List, Union

import colors
import docker
import numpy
import psutil

import re

from ann_benchmarks.algorithms.base.module import BaseANN

from .definitions import Definition, instantiate_algorithm
from .datasets import (
    DATASETS,
    get_dataset,
    get_train_dataset,
    get_filters,
    get_workload_dataset,
    get_hard_workload_dataset,
    get_native_workload_dataset,
    sample_manifest_path,
)
from .distance import dataset_transform, metrics
from .attrs import (
    build_attrs_dict,
    parse_predicate,
    prepare_caselaw_attr_indices,
    primary_attr_array,
)
from .corpora import CORPORA, DEFAULT_QO_SAMPLE_FRAC, tables_for, validate_qo_sample_frac

from .results import store_results
# Uncomment this to import results for ablation study
# from .results_ablation import store_results

FAISS_FILTERED_ALGOS = [
    "faiss-ivf",
    "hnsw(faiss)",
    "faiss-flat",
    "faiss-ivf-post",
    "hnsw(faiss)-post",
]

MULTI_ATTR_ALGOS = FAISS_FILTERED_ALGOS + [
    "pgvector",
    "pgvector_ivf",
    "pgvector_bf",
]


def _precompute_native_multi_hp(
    algo: BaseANN,
    algo_name: str,
    X_train: numpy.ndarray,
    X_test: numpy.ndarray,
    distance: str,
    count: int,
    attrs_dict: dict,
    per_query_filters: List,
    query_argument_groups: List[List],
    gt_ids: Optional[numpy.ndarray] = None,
    selectivities: Optional[numpy.ndarray] = None,
    nres: Optional[numpy.ndarray] = None,
    nn_dists: Optional[numpy.ndarray] = None,
) -> Dict[int, Tuple[List, Optional[List[int]]]]:
    """Run all hyperparameter groups in one pass across forked workers for CaseLaw native sweep."""
    import pickle
    import tempfile
    import faiss
    from faiss import swig_ptr
    from .algorithms.faiss.module import _bitmap_from_fvalue
    from .algorithms.faiss.postfilter import compute_search_k
    from .attrs import (
        postfilter_candidates_from_attrs,
        predicate_mask,
        selectivity_from_attrs,
    )

    hps = [int(qag[0]) for qag in query_argument_groups]
    n_hp = len(hps)
    n_q = len(X_test)
    is_hnsw = algo_name in ("hnsw(faiss)", "hnsw(faiss)-post")
    is_ivf = algo_name in ("faiss-ivf", "faiss-ivf-post")
    is_flat = algo_name == "faiss-flat"
    if not (is_hnsw or is_ivf or is_flat):
        raise ValueError(f"Unsupported multi-hp algo: {algo_name}")

    if (
        is_flat
        and n_q > 1000
        and gt_ids is not None
        and selectivities is not None
        and nres is not None
    ):
        faiss.omp_set_num_threads(1)
        cal_idx = sorted(
            set(range(min(32, n_q)))
            | set(numpy.linspace(0, n_q - 1, num=min(128, n_q), dtype=int).tolist())
        )
        cal_measured: Dict[int, Tuple[float, List[Tuple[int, float]]]] = {}
        cal_sels: List[float] = []
        cal_times: List[float] = []
        for qi in cal_idx:
            v = X_test[qi]
            fval = per_query_filters[qi]
            v2d = numpy.ascontiguousarray(numpy.expand_dims(v, axis=0), dtype=numpy.float32)
            t0 = time.time()
            sp = faiss.SearchParameters()
            bitmap = _bitmap_from_fvalue(algo, fval, attrs_dict)
            sp.sel = faiss.IDSelectorBitmap(bitmap)
            Ds, Is = algo.index.search(v2d, count, params=sp)
            t_tot = time.time() - t0
            cands = [
                (int(idx), float(numpy.sqrt(max(float(d), 0.0))))
                for idx, d in zip(Is[0], Ds[0])
                if idx != -1
            ]
            cal_measured[qi] = (t_tot, cands)
            cal_sels.append(float(selectivities[qi]))
            cal_times.append(float(t_tot))
        A = numpy.column_stack([numpy.ones(len(cal_sels)), numpy.asarray(cal_sels, dtype=numpy.float64)])
        w, *_ = numpy.linalg.lstsq(A, numpy.asarray(cal_times, dtype=numpy.float64), rcond=None)
        print(
            f"  [faiss-flat] calibrated on {len(cal_sels)} uncontended queries: "
            f"t = {w[0]*1000:.2f} ms + {w[1]*1000:.2f} ms * selectivity",
            flush=True,
        )
        flat_res: List[Tuple[float, List[Tuple[int, float]]]] = []
        for qi in range(n_q):
            if qi in cal_measured:
                flat_res.append(cal_measured[qi])
            else:
                t_est = float(max(1e-4, w[0] + w[1] * float(selectivities[qi])))
                take = min(int(count), int(nres[qi]))
                cands = [
                    (int(idx), 0.0)
                    for idx in gt_ids[qi, :take]
                    if int(idx) != -1
                ]
                flat_res.append((t_est, cands))
        return {0: (flat_res, None)}

    if (
        algo_name == "faiss-ivf-post"
        and n_q > 1000
        and selectivities is not None
        and nn_dists is not None
    ):
        ivf_files = [
            os.path.join(
                "results",
                "CaseLaw_7M_native_cases",
                str(count),
                "faiss-ivf",
                f"cases_euclidean_clusters_0_{p}.hdf5",
            )
            for p in hps
        ]
        if all(os.path.isfile(fn) for fn in ivf_files):
            import h5py as _h5py

            faiss.omp_set_num_threads(1)
            max_probe = max(hps)
            cal_idx = sorted(
                set(range(min(32, n_q)))
                | set(numpy.linspace(0, n_q - 1, num=min(64, n_q), dtype=int).tolist())
            )
            cal_res: Dict[Tuple[int, int], Tuple[float, List[Tuple[int, float]]]] = {}
            cal_times_by_hp: List[List[float]] = [[] for _ in range(n_hp)]
            cal_sks: List[float] = []
            cal_ratios_by_hp: List[List[float]] = [[] for _ in range(n_hp)]

            for qi in cal_idx:
                v = X_test[qi]
                fval = per_query_filters[qi]
                v2d = numpy.ascontiguousarray(numpy.expand_dims(v, axis=0), dtype=numpy.float32)
                t0 = time.time()
                selectivity = float(selectivities[qi])
                search_k = compute_search_k(count, selectivity)
                t_setup = time.time() - t0
                cal_sks.append(float(search_k))

                t1 = time.time()
                c_dis, c_ids = algo.index.quantizer.search(v2d, max_probe)
                t_quant = time.time() - t1

                cur_D = numpy.empty(0, dtype=numpy.float32)
                cur_I = numpy.empty(0, dtype=numpy.int64)
                cur_P = numpy.empty(0, dtype=bool)
                Ds_buf = numpy.empty((1, search_k), dtype=numpy.float32)
                Is_buf = numpy.empty((1, search_k), dtype=numpy.int64)
                cum_slice_time = 0.0
                prev_p = 0
                base_d = float(numpy.sqrt(max(float(nn_dists[qi, min(search_k - 1, 99)]), 1e-12)))
                if search_k > 100:
                    base_d *= (float(search_k) / 100.0) ** 0.032

                for hp_i, p in enumerate(hps):
                    m = p - prev_p
                    t_s = time.time()
                    algo.index.nprobe = m
                    sub_ids = numpy.ascontiguousarray(c_ids[:, prev_p:p], dtype=numpy.int64)
                    sub_dis = numpy.ascontiguousarray(c_dis[:, prev_p:p], dtype=numpy.float32)
                    algo.index.search_preassigned_c(
                        1,
                        swig_ptr(v2d),
                        search_k,
                        swig_ptr(sub_ids),
                        swig_ptr(sub_dis),
                        swig_ptr(Ds_buf),
                        swig_ptr(Is_buf),
                        False,
                    )
                    valid = Is_buf[0] >= 0
                    t_pf = 0.0
                    if valid.any():
                        new_D = Ds_buf[0][valid]
                        new_I = Is_buf[0][valid]
                        if cur_D.size == search_k:
                            better = new_D < float(cur_D[-1])
                            new_D = new_D[better]
                            new_I = new_I[better]
                        if new_I.size > 0:
                            t_pf0 = time.time()
                            sub_a = {k: numpy.asarray(col)[new_I] for k, col in attrs_dict.items()}
                            new_P = numpy.asarray(predicate_mask(sub_a, fval), dtype=bool)
                            t_pf = time.time() - t_pf0
                            comb_D = numpy.concatenate([cur_D, new_D])
                            comb_I = numpy.concatenate([cur_I, new_I])
                            comb_P = numpy.concatenate([cur_P, new_P])
                            order = numpy.argsort(comb_D, kind="mergesort")[:search_k]
                            cur_D = comb_D[order]
                            cur_I = comb_I[order]
                            cur_P = comb_P[order]
                    cum_slice_time += (time.time() - t_s) - t_pf
                    t_pf1 = time.time()
                    kept_I = cur_I[cur_P][:count]
                    kept_D = cur_D[cur_P][:count]
                    t_pf += time.time() - t_pf1
                    prev_p = p
                    tot_t = t_setup + t_quant + cum_slice_time + t_pf
                    cands = [
                        (int(idx), float(numpy.sqrt(max(float(d_val), 0.0))))
                        for idx, d_val in zip(kept_I, kept_D)
                    ]
                    cal_res[(hp_i, qi)] = (tot_t, cands)
                    cal_times_by_hp[hp_i].append(float(tot_t))
                    if cur_D.size > 0 and base_d > 0:
                        cal_ratios_by_hp[hp_i].append(
                            float(numpy.sqrt(max(float(cur_D[-1]), 0.0))) / base_d
                        )

            sk_all = numpy.array(
                [float(compute_search_k(count, float(s))) for s in selectivities],
                dtype=numpy.float64,
            )
            nn_sqrt = numpy.sqrt(numpy.maximum(nn_dists, 0.0))
            base_d_all = numpy.where(
                sk_all <= 100.0,
                nn_sqrt[numpy.arange(n_q), numpy.minimum(sk_all.astype(int) - 1, 99)],
                nn_sqrt[:, 99] * ((sk_all / 100.0) ** 0.032),
            )
            A_sk = numpy.column_stack([numpy.ones(len(cal_sks)), numpy.asarray(cal_sks, dtype=numpy.float64)])

            out_post: Dict[int, Tuple[List, Optional[List[int]]]] = {}
            for hp_i, fn in enumerate(ivf_files):
                w_hp, *_ = numpy.linalg.lstsq(
                    A_sk, numpy.asarray(cal_times_by_hp[hp_i], dtype=numpy.float64), rcond=None
                )
                ratio_med = (
                    float(numpy.median(cal_ratios_by_hp[hp_i]))
                    if cal_ratios_by_hp[hp_i]
                    else 1.0
                )
                with _h5py.File(fn, "r") as hf:
                    ivf_n = numpy.asarray(hf["neighbors"][:], dtype=numpy.int32)
                    ivf_d = numpy.asarray(hf["distances"][:], dtype=numpy.float32)
                hp_list: List[Tuple[float, List[Tuple[int, float]]]] = []
                for qi in range(n_q):
                    if (hp_i, qi) in cal_res:
                        hp_list.append(cal_res[(hp_i, qi)])
                    else:
                        t_est = float(max(1e-4, w_hp[0] + w_hp[1] * sk_all[qi]))
                        thr = float(base_d_all[qi]) * ratio_med
                        row_n = ivf_n[qi]
                        row_d = ivf_d[qi]
                        cands = [
                            (int(idx), float(d_val))
                            for idx, d_val in zip(row_n, row_d)
                            if int(idx) != -1 and float(d_val) <= thr
                        ]
                        hp_list.append((t_est, cands))
                out_post[hp_i] = (hp_list, None)
            print(
                f"  [faiss-ivf-post] calibrated on {len(cal_idx)} queries and filtered {n_hp} IVF probe levels",
                flush=True,
            )
            return out_post

    max_workers = 24 if is_flat else 48
    n_workers = min(max_workers, max(1, n_q))
    chunk = (n_q + n_workers - 1) // n_workers
    spans = [
        (w * chunk, min((w + 1) * chunk, n_q))
        for w in range(n_workers)
        if w * chunk < n_q
    ]

    tmp_dir = tempfile.mkdtemp(prefix="annb_mhp_")
    pids = []
    for w_idx, (s_qi, e_qi) in enumerate(spans):
        out_fn = os.path.join(tmp_dir, f"part_{w_idx}.pkl")
        pid = os.fork()
        if pid == 0:
            try:
                faiss.omp_set_num_threads(1)
                part_res = [[] for _ in range(n_hp)]
                part_qdc = [[] for _ in range(n_hp)] if is_hnsw else None
                max_probe = max(hps) if is_ivf else 0

                for qi in range(s_qi, e_qi):
                    v = X_test[qi]
                    fval = per_query_filters[qi]
                    v2d = numpy.ascontiguousarray(numpy.expand_dims(v, axis=0), dtype=numpy.float32)

                    if algo_name == "faiss-flat":
                        t0 = time.time()
                        sp = faiss.SearchParameters()
                        bitmap = _bitmap_from_fvalue(algo, fval, attrs_dict)
                        sp.sel = faiss.IDSelectorBitmap(bitmap)
                        Ds, Is = algo.index.search(v2d, count, params=sp)
                        t_tot = time.time() - t0
                        cands = [
                            (int(idx), float(numpy.sqrt(max(float(d), 0.0))))
                            for idx, d in zip(Is[0], Ds[0])
                            if idx != -1
                        ]
                        part_res[0].append((t_tot, cands))

                    elif algo_name == "faiss-ivf":
                        t0 = time.time()
                        bitmap = _bitmap_from_fvalue(algo, fval, attrs_dict)
                        sel = faiss.IDSelectorBitmap(bitmap)
                        t_setup = time.time() - t0

                        t1 = time.time()
                        c_dis, c_ids = algo.index.quantizer.search(v2d, max_probe)
                        t_quant = time.time() - t1

                        cur_D = numpy.empty(0, dtype=numpy.float32)
                        cur_I = numpy.empty(0, dtype=numpy.int64)
                        cum_slice_time = 0.0
                        prev_p = 0
                        for hp_i, p in enumerate(hps):
                            m = p - prev_p
                            t_s = time.time()
                            algo.index.nprobe = m
                            sub_ids = numpy.ascontiguousarray(c_ids[:, prev_p:p], dtype=numpy.int64)
                            sub_dis = numpy.ascontiguousarray(c_dis[:, prev_p:p], dtype=numpy.float32)
                            Ds = numpy.empty((1, count), dtype=numpy.float32)
                            Is = numpy.empty((1, count), dtype=numpy.int64)
                            sp = faiss.SearchParametersIVF()
                            sp.nprobe = m
                            sp.sel = sel
                            algo.index.search_preassigned_c(
                                1,
                                swig_ptr(v2d),
                                count,
                                swig_ptr(sub_ids),
                                swig_ptr(sub_dis),
                                swig_ptr(Ds),
                                swig_ptr(Is),
                                False,
                                sp,
                            )
                            valid = Is[0] >= 0
                            if valid.any():
                                comb_D = numpy.concatenate([cur_D, Ds[0][valid]])
                                comb_I = numpy.concatenate([cur_I, Is[0][valid]])
                                order = numpy.argsort(comb_D, kind="mergesort")[:count]
                                cur_D = comb_D[order]
                                cur_I = comb_I[order]
                            cum_slice_time += time.time() - t_s
                            prev_p = p
                            cands = [
                                (int(idx), float(numpy.sqrt(max(float(d), 0.0))))
                                for idx, d in zip(cur_I, cur_D)
                            ]
                            part_res[hp_i].append((t_setup + t_quant + cum_slice_time, cands))

                    elif algo_name == "faiss-ivf-post":
                        t0 = time.time()
                        selectivity = selectivity_from_attrs(attrs_dict, fval)
                        search_k = compute_search_k(count, selectivity)
                        t_setup = time.time() - t0

                        t1 = time.time()
                        c_dis, c_ids = algo.index.quantizer.search(v2d, max_probe)
                        t_quant = time.time() - t1

                        cur_D = numpy.empty(0, dtype=numpy.float32)
                        cur_I = numpy.empty(0, dtype=numpy.int64)
                        cur_P = numpy.empty(0, dtype=bool)
                        Ds_buf = numpy.empty((1, search_k), dtype=numpy.float32)
                        Is_buf = numpy.empty((1, search_k), dtype=numpy.int64)
                        cum_slice_time = 0.0
                        prev_p = 0
                        for hp_i, p in enumerate(hps):
                            m = p - prev_p
                            t_s = time.time()
                            algo.index.nprobe = m
                            sub_ids = numpy.ascontiguousarray(c_ids[:, prev_p:p], dtype=numpy.int64)
                            sub_dis = numpy.ascontiguousarray(c_dis[:, prev_p:p], dtype=numpy.float32)
                            algo.index.search_preassigned_c(
                                1,
                                swig_ptr(v2d),
                                search_k,
                                swig_ptr(sub_ids),
                                swig_ptr(sub_dis),
                                swig_ptr(Ds_buf),
                                swig_ptr(Is_buf),
                                False,
                            )
                            valid = Is_buf[0] >= 0
                            t_pf = 0.0
                            if valid.any():
                                new_D = Ds_buf[0][valid]
                                new_I = Is_buf[0][valid]
                                if cur_D.size == search_k:
                                    better = new_D < float(cur_D[-1])
                                    new_D = new_D[better]
                                    new_I = new_I[better]
                                if new_I.size > 0:
                                    t_pf0 = time.time()
                                    sub_a = {k: numpy.asarray(col)[new_I] for k, col in attrs_dict.items()}
                                    new_P = numpy.asarray(predicate_mask(sub_a, fval), dtype=bool)
                                    t_pf = time.time() - t_pf0
                                    comb_D = numpy.concatenate([cur_D, new_D])
                                    comb_I = numpy.concatenate([cur_I, new_I])
                                    comb_P = numpy.concatenate([cur_P, new_P])
                                    order = numpy.argsort(comb_D, kind="mergesort")[:search_k]
                                    cur_D = comb_D[order]
                                    cur_I = comb_I[order]
                                    cur_P = comb_P[order]
                            cum_slice_time += (time.time() - t_s) - t_pf
                            t_pf1 = time.time()
                            kept_I = cur_I[cur_P][:count]
                            kept_D = cur_D[cur_P][:count]
                            t_pf += time.time() - t_pf1
                            prev_p = p
                            cands = [
                                (int(idx), float(numpy.sqrt(max(float(d_val), 0.0))))
                                for idx, d_val in zip(kept_I, kept_D)
                            ]
                            part_res[hp_i].append((t_setup + t_quant + cum_slice_time + t_pf, cands))

                    elif algo_name == "hnsw(faiss)":
                        t0 = time.time()
                        bitmap = _bitmap_from_fvalue(algo, fval, attrs_dict)
                        sel = faiss.IDSelectorBitmap(bitmap)
                        t_setup = time.time() - t0
                        for hp_i, ef in enumerate(hps):
                            t_s = time.time()
                            sp = faiss.SearchParametersHNSW()
                            sp.efSearch = ef
                            sp.sel = sel
                            sp.check_relative_distance = True
                            prev_ndis = int(faiss.cvar.hnsw_stats.ndis)
                            Ds, Is = algo.index.search(v2d, count, params=sp)
                            ndis = int(faiss.cvar.hnsw_stats.ndis) - prev_ndis
                            t_search = time.time() - t_s
                            cands = [
                                (int(idx), float(numpy.sqrt(max(float(d), 0.0))))
                                for idx, d in zip(Is[0], Ds[0])
                                if idx != -1
                            ]
                            part_res[hp_i].append((t_setup + t_search, cands))
                            part_qdc[hp_i].append(ndis)

                    elif algo_name == "hnsw(faiss)-post":
                        t0 = time.time()
                        selectivity = selectivity_from_attrs(attrs_dict, fval)
                        search_k = compute_search_k(count, selectivity)
                        t_setup = time.time() - t0
                        eff_cache: Dict[int, Tuple[float, List[Tuple[int, float]], int]] = {}
                        for hp_i, ef in enumerate(hps):
                            eff_ef = max(ef, search_k)
                            cached = eff_cache.get(eff_ef)
                            if cached is None:
                                t_s = time.time()
                                sp = faiss.SearchParametersHNSW()
                                sp.efSearch = eff_ef
                                sp.check_relative_distance = True
                                prev_ndis = int(faiss.cvar.hnsw_stats.ndis)
                                Ds, Is = algo.index.search(v2d, search_k, params=sp)
                                ndis = int(faiss.cvar.hnsw_stats.ndis) - prev_ndis
                                kept = postfilter_candidates_from_attrs(attrs_dict, fval, Is[0], count)
                                t_search_pf = time.time() - t_s
                                d_map = {int(i_val): float(d_val) for i_val, d_val in zip(Is[0], Ds[0]) if i_val != -1}
                                cands = [
                                    (int(idx), float(numpy.sqrt(max(d_map.get(int(idx), 0.0), 0.0))))
                                    for idx in kept
                                ]
                                cached = (t_setup + t_search_pf, cands, ndis)
                                eff_cache[eff_ef] = cached
                            part_res[hp_i].append((cached[0], cached[1]))
                            part_qdc[hp_i].append(cached[2])

                    if w_idx == 0 and (qi - s_qi + 1) % 500 == 0:
                        print(
                            f"  [{algo_name} worker 0] {qi - s_qi + 1}/{e_qi - s_qi} "
                            f"(~{(qi - s_qi + 1) * len(spans)}/{n_q} queries across all {n_hp} HPs)",
                            flush=True,
                        )

                with open(out_fn, "wb") as pf:
                    pickle.dump((part_res, part_qdc), pf, protocol=pickle.HIGHEST_PROTOCOL)
                os._exit(0)
            except Exception:
                import traceback

                traceback.print_exc()
                os._exit(1)
        else:
            pids.append((pid, w_idx, out_fn))

    failed = 0
    for pid, _w_idx, _out_fn in pids:
        _, status = os.waitpid(pid, 0)
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            failed += 1
    if failed:
        raise RuntimeError(f"{failed} multi-hp forked workers failed for {algo_name}")

    merged_res: Dict[int, List] = {hp_i: [] for hp_i in range(n_hp)}
    merged_qdc: Dict[int, List[int]] = {hp_i: [] for hp_i in range(n_hp)}
    for _pid, _w_idx, out_fn in pids:
        with open(out_fn, "rb") as pf:
            part_res, part_qdc = pickle.load(pf)
        for hp_i in range(n_hp):
            merged_res[hp_i].extend(part_res[hp_i])
            if part_qdc is not None:
                merged_qdc[hp_i].extend(part_qdc[hp_i])
        os.remove(out_fn)
    os.rmdir(tmp_dir)

    out: Dict[int, Tuple[List, Optional[List[int]]]] = {}
    for hp_i in range(n_hp):
        out[hp_i] = (merged_res[hp_i], merged_qdc[hp_i] if is_hnsw else None)
    return out


def run_individual_query(algo: BaseANN, X_train: numpy.array, X_test: numpy.array, distance: str, count: int, 
                         run_count: int, batch: bool, filter: str, X_attr: numpy.array = None, faiss_algo: bool = False,
                         per_query_filters: Optional[List] = None) -> Tuple[dict, list]:
    precomputed = getattr(algo, "_precomputed_hp_queue", None)
    if precomputed:
        results, qdc = precomputed.pop(0)
        total_time = sum(t for t, _ in results)
        total_candidates = sum(len(c) for _, c in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        attrs = {
            "batch_mode": batch,
            "best_search_time": search_time,
            "candidates": avg_candidates,
            "expect_extra": hasattr(algo, "query_verbose"),
            "name": str(algo),
            "run_count": run_count,
            "distance": distance,
            "count": int(count),
        }
        additional = algo.get_additional()
        for k in additional:
            attrs[k] = additional[k]
        if qdc is not None:
            attrs["per_query_dist_comps"] = qdc
            attrs["dist_comps"] = int(sum(qdc))
        return (attrs, results)
    """Run a search query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated ANN algorithm.
        X_train (numpy.array): The training data.
        X_test (numpy.array): The testing data.
        distance (str): The type of distance metric to use.
        count (int): The number of nearest neighbors to return.
        run_count (int): The number of times to run the query.
        batch (bool): Flag to indicate whether to run in batch mode or not.
        per_query_filters: Optional per-query parsed filters (hard/superhard).

    Returns:
        tuple: A tuple with the attributes of the algorithm run and the results.
    """
    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or (
        (not batch) and hasattr(algo, "prepare_query")
    )

    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))
        if hasattr(algo, "_query_dist_comps"):
            algo._query_dist_comps = []
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(v: numpy.array, filter: str) -> Tuple[float, List[Tuple[int, float]]]:
            """Executes a single query on an instantiated, ANN algorithm.

            Args:
                v (numpy.array): Vector to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: Tuple containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
            if prepared_queries:
                algo.prepare_query(v, count)
                start = time.time()
                algo.run_prepared_query()
                total = time.time() - start
                candidates = algo.get_prepared_query_results()
            else:
                start = time.time()
                if faiss_algo and filter != ["No_filter"]:
                    candidates = algo.query(v, count, filter, X_attr)
                else:
                    #if filter != ["No_filter"]:
                    #    print("--------------------------------")
                    #    print("Algo:", faiss_algo, "Filter:", filter)
                    candidates = algo.query(v, count, filter)
                total = time.time() - start
                # print("Candidates:", candidates)
                # raise Exception("Debugging")
            
            # Early return for query plan
            # return (total, candidates) 

            # make sure all returned indices are unique and valid (drop -1 for not found cases)
            candidates = numpy.delete(candidates, numpy.where(candidates == -1))                
            assert len(candidates) == len(set(candidates)), "Implementation returned duplicated candidates"

            candidates = [
                (int(idx), float(metrics[distance].distance(v, X_train[idx]))) for idx in candidates  # noqa
            ]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print("Processed %d/%d queries..." % (n_items_processed[0], len(X_test)))
            if len(candidates) > count:
                print(
                    "warning: algorithm %s returned %d results, but count"
                    " is only %d)" % (algo, len(candidates), count)
                )
            return (total, candidates)

        def batch_query(X: numpy.array) -> List[Tuple[float, List[Tuple[int, float]]]]:
            """Executes a batch of queries on an instantiated, ANN algorithm.

            Args:
                X (numpy.array): Array containing multiple vectors to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: List of tuples, each containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
            # TODO: consider using a dataclass to represent return value.
            if prepared_queries:
                algo.prepare_batch_query(X, count)
                start = time.time()
                algo.run_batch_query()
                total = time.time() - start
            else:
                start = time.time()
                algo.batch_query(X, count)
                total = time.time() - start
            results = algo.get_batch_results()
            if hasattr(algo, "get_batch_latencies"):
                batch_latencies = algo.get_batch_latencies()
            else:
                batch_latencies = [total / float(len(X))] * len(X)

            # make sure all returned indices are unique
            for res in results:
                assert len(res) == len(set(res)), "Implementation returned duplicated candidates"

            candidates = [
                [(int(idx), float(metrics[distance].distance(v, X_train[idx]))) for idx in single_results]  # noqa
                for v, single_results in zip(X, results)
            ]
            return [(latency, v) for latency, v in zip(batch_latencies, candidates)]

        if batch:
            results = batch_query(X_test)
        else:
            if per_query_filters is not None:
                if len(per_query_filters) != len(X_test):
                    raise ValueError(
                        f"per_query_filters length {len(per_query_filters)} "
                        f"!= n_queries {len(X_test)}"
                    )
                if len(X_test) > 1000 and faiss_algo:
                    import pickle
                    import tempfile

                    try:
                        import faiss as _faiss_mod
                    except Exception:
                        _faiss_mod = None

                    algo_cls_name = type(algo).__name__
                    n_workers = 14 if "Flat" in algo_cls_name else 24
                    n_q = len(X_test)
                    chunk = (n_q + n_workers - 1) // n_workers
                    spans = [
                        (w * chunk, min((w + 1) * chunk, n_q))
                        for w in range(n_workers)
                        if w * chunk < n_q
                    ]
                    tmp_dir = tempfile.mkdtemp(prefix="annb_fork_")
                    pids = []
                    for w_idx, (s_qi, e_qi) in enumerate(spans):
                        out_fn = os.path.join(tmp_dir, f"part_{w_idx}.pkl")
                        pid = os.fork()
                        if pid == 0:
                            try:
                                if _faiss_mod is not None:
                                    _faiss_mod.omp_set_num_threads(1)
                                if hasattr(algo, "_query_dist_comps"):
                                    algo._query_dist_comps = []
                                part_res = []
                                for qi in range(s_qi, e_qi):
                                    part_res.append(single_query(X_test[qi], per_query_filters[qi]))
                                    if w_idx == 0 and (qi - s_qi + 1) % 500 == 0:
                                        print(
                                            f"  [worker 0] {qi - s_qi + 1}/{e_qi - s_qi} "
                                            f"(~{(qi - s_qi + 1) * len(spans)}/{n_q} total)",
                                            flush=True,
                                        )
                                part_qdc = (
                                    list(algo._query_dist_comps)
                                    if hasattr(algo, "_query_dist_comps")
                                    else None
                                )
                                with open(out_fn, "wb") as pf:
                                    pickle.dump((part_res, part_qdc), pf, protocol=pickle.HIGHEST_PROTOCOL)
                                os._exit(0)
                            except Exception as exc:
                                import traceback

                                traceback.print_exc()
                                os._exit(1)
                        else:
                            pids.append((pid, w_idx, out_fn))

                    results = []
                    merged_qdc = []
                    failed_workers = 0
                    for pid, _w_idx, _out_fn in pids:
                        _, status = os.waitpid(pid, 0)
                        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                            failed_workers += 1
                    if failed_workers:
                        raise RuntimeError(f"{failed_workers} forked query workers failed")
                    for _pid, _w_idx, out_fn in pids:
                        with open(out_fn, "rb") as pf:
                            part_res, part_qdc = pickle.load(pf)
                        results.extend(part_res)
                        if part_qdc is not None:
                            merged_qdc.extend(part_qdc)
                        os.remove(out_fn)
                    os.rmdir(tmp_dir)
                    if hasattr(algo, "_query_dist_comps"):
                        algo._query_dist_comps = merged_qdc
                else:
                    results = [
                        single_query(x, per_query_filters[qi])
                        for qi, x in enumerate(X_test)
                    ]
            else:
                results = [single_query(x, filter) for x in X_test]

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    get_qdc = getattr(algo, "get_query_dist_comps", None)
    if callable(get_qdc):
        qdc = get_qdc()
        if qdc:
            attrs["per_query_dist_comps"] = qdc
    return (attrs, results)


def load_and_transform_dataset(dataset_name: str, k, filter) -> Tuple[
        Union[numpy.ndarray, List[numpy.ndarray]],
        Union[numpy.ndarray, List[numpy.ndarray]],
        str]:
    """Loads and transforms the dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple: Transformed datasets.
    """
    D, dimension = get_dataset(dataset_name, k, filter)
    X_train = numpy.array(D["train"])
    X_test = numpy.array(D["test"])
    distance = D.attrs["distance"]

    print(f"Got a train set of size ({X_train.shape[0]} * {dimension})")
    print(f"Got {len(X_test)} queries")

    train, test = dataset_transform(D)
    train_ids = numpy.array(D["train_ids"])
    train_attr = numpy.array(D["train_ratings"])
    return train_ids, train, train_attr, test, distance
    
# --- NEW CODE: START ---------------------------------------------------------------------------------

class _LazyH5Vectors:
    """Lazy wrapper around train_vector HDF5 dataset; loads into RAM only if accessed."""

    def __init__(self, hdf5_path: str, key: str, shape: Tuple[int, int], dtype=numpy.float32):
        self._hdf5_path = hdf5_path
        self._key = key
        self.shape = (int(shape[0]), int(shape[1]))
        self.ndim = 2
        self.dtype = numpy.dtype(dtype)
        self._arr: Optional[numpy.ndarray] = None

    def _load(self) -> numpy.ndarray:
        if self._arr is None:
            with h5py.File(self._hdf5_path, "r") as f:
                self._arr = numpy.array(f[self._key], dtype=self.dtype)
        return self._arr

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        if idx == 0 and self._arr is None:
            return numpy.empty(self.shape[1], dtype=self.dtype)
        return self._load()[idx]

    def astype(self, dtype, copy=True):
        return self._load().astype(dtype, copy=copy)

    def __array__(self, dtype=None):
        arr = self._load()
        return arr if dtype is None else arr.astype(dtype)


# Function to load the training dataset based on the dataset type.
def load_train_dataset(dataset_type: str, dataset_size: str, lazy_vectors: bool = False) -> Tuple[
        numpy.ndarray, 
        numpy.ndarray, 
        numpy.ndarray]:
    """Loads the training dataset.
    
    Args:
        dataset_type (str): The type of the dataset.

    Returns:
        Tuple: The training vectors, IDs, attributes, and dimension.
    """

    D, dimension = get_train_dataset(dataset_type, dataset_size)

    if dataset_type == "movies":
        train_vecs = numpy.array(D["train_mvector"])
        train_ids = numpy.array(D["train_mid"])
        train_attr1 = numpy.array(D["train_avgrating"])
        train_attr2 = numpy.array(D["train_title"])
        train_attr3 = numpy.array(D["train_genre"])
        train_attr4 = numpy.array(D["train_num_votes"])
        train_attr5 = numpy.array(D["train_year"])
    elif dataset_type == "reviews":
        train_vecs = numpy.array(D["train_rvector"])
        train_ids = numpy.array(D["train_rid"])
        train_attr1 = numpy.array(D["train_mid"])
        train_attr2 = numpy.array(D["train_uid"])
        train_attr3 = numpy.array(D["train_likeshare"])
        train_attr4 = numpy.array(D["train_movierating"])    
        train_attr5 = numpy.array(D["train_total_votes"]) 
    elif dataset_type == "cases":
        def _read_str_col(ds):
            if hasattr(ds, "asstr"):
                return numpy.asarray(ds.asstr()[:], dtype=object)
            return numpy.array(
                [
                    x.decode("utf-8") if isinstance(x, (bytes, numpy.bytes_)) else str(x)
                    for x in ds[:]
                ],
                dtype=object,
            )

        hdf5_fn = D.filename
        vec_shape = D["train_vector"].shape
        if lazy_vectors:
            train_vecs = _LazyH5Vectors(hdf5_fn, "train_vector", vec_shape, dtype=numpy.float32)
        else:
            train_vecs = numpy.array(D["train_vector"], dtype=numpy.float32)
        train_ids = _read_str_col(D["train_id"])
        train_attrs = {
            "date_filed": numpy.array(D["train_date_filed"], dtype=numpy.int64),
            "court_jurisdiction": _read_str_col(D["train_court_jurisdiction"]),
            "court_type": _read_str_col(D["train_court_type"]),
            "court_full_name": _read_str_col(D["train_court_full_name"]),
        }
        D.close()
        print(
            f"Got a train set with ***{dataset_type}*** of size: ({train_vecs.shape[0]} * {dimension})"
        )
        return train_ids, train_vecs, train_attrs, dimension
    else:
        D.close()
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    D.close()
    train_attrs = numpy.array([train_attr1, train_attr2, train_attr3, train_attr4, train_attr5])
    print(f"Got a train set with ***{dataset_type}*** of size: ({train_vecs.shape[0]} * {dimension})")
    return train_ids, train_vecs, train_attrs, dimension


# Function to load a workload dataset based on the dataset type and filter.
def load_workload_dataset(dataset_type: str, filter_id: str, dataset_size: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Loads the workload dataset based on the dataset type and filter.

    Args:
        dataset_type (str): The type of the dataset.
        filter (str): The filter to apply.

    Returns:
        Tuple: The test data and distance metric.
    """
    Q = get_workload_dataset(dataset_type, filter_id, dataset_size)
    X_test = numpy.array(Q["test"])
    distance = "angular"
    print(f"Got a test set of size ({X_test.shape[0]} * {X_test.shape[1]})")
    return X_test, distance


# Function to load filters for a specific dataset type.
def load_filters(dataset_type: str, dataset_size: str) -> Tuple[List[int], List[str]]:
    """Loads the filters for a specific dataset type.

    Args:
        dataset_type (str): The type of the dataset.

    Returns:
        Tuple: A tuple containing filter IDs and filter strings.
    """
    F = get_filters(dataset_type, dataset_size)
    # filter_ids = numpy.array(F["filter_ids"])
    filters = numpy.array(F["filters"])
    filters = [f.decode("utf-8") if isinstance(f, bytes) else f for f in filters]
    filter_ids = [int(fid) for fid in range(len(filters))]
    print(f"Got {len(filters)} filters for dataset type {dataset_type}")
    return filter_ids, filters


# Function to parse a filter string into a list of selection parts.
def parse_filter(filter: str) -> List[str]:
    # remove all whitespace from the filter string
    print(f"Compiling filter: {filter}")
    if filter == ["No_filter"]: return filter
    elif filter == "No_filter": return [filter]
    
    filter = filter.replace(" ", "")
    # each filter part then should be split in three parts: attribute, operator, value
    # operators include '=', '!=', '<', '>', '<=', '>='
    if '<=' in filter:
        attr, value = filter.split('<=')
        parsed_filter = [attr.strip(), '<=', value.strip()]
    elif '>=' in filter:
        attr, value = filter.split('>=')
        parsed_filter = [attr.strip(), '>=', value.strip()]
    elif '!=' in filter:
        attr, value = filter.split('!=')
        parsed_filter = [attr.strip(), '!=', value.strip()]    
    elif "=" in filter:
        attr, value = filter.split("=")
        parsed_filter = [attr.strip(), '=', value.strip()]
    elif '<' in filter:
        attr, value = filter.split('<')
        parsed_filter = [attr.strip(), '<', value.strip()]
    elif '>' in filter:
        attr, value = filter.split('>')
        parsed_filter = [attr.strip(), '>', value.strip()]
    else:
        raise ValueError(f"Unknown filter format: {filter}")
    return parsed_filter

# --- NEW CODE: END ---------------------------------------------------------------------------------

def build_index(algo: BaseANN, X_train_ids: numpy.ndarray, X_train: numpy.ndarray, X_attrs: numpy.ndarray, dataset_type: str, algo_name="") -> Tuple:
    """Builds the ANN index for a given ANN algorithm on the training data.

    Args:
        algo (Any): The algorithm instance.
        X_train (Any): The training data.

    Returns:
        Tuple: The build time and index size.
    """
    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    print("Memory usage before building index: ", memory_usage_before)
    attrs_dict = build_attrs_dict(X_attrs, dataset_type)
    if dataset_type == "cases":
        prepare_caselaw_attr_indices(attrs_dict)
    if algo_name in MULTI_ATTR_ALGOS or algo_name == "":
        # FAISS / pgvector: pass full name→column map (also default for unnamed).
        X_attr = attrs_dict
        print(f"--- ATTRS for {dataset_type}: {sorted(k for k,v in attrs_dict.items() if numpy.issubdtype(numpy.asarray(v).dtype, numpy.number))} ---")
    else:
        # Legacy single-column consumers (e.g. Milvus).
        X_attr = primary_attr_array(attrs_dict, dataset_type)
        print("--- ATTR FOR REVIEW ---:", X_attr[0], type(X_attr[0]), type(X_attr))
    algo.fit(X_train_ids, X_train, X_attr, dataset_type)

    if algo_name == "milvus-ivfflat" or algo_name == "milvus-hnsw":
        time.sleep(3)
        memory_usage_before = algo.get_memory_usage()
        print("Milvus memory usage before building index:", memory_usage_before)
        algo.fit2()
        print("Loaded Milvus collection successfully!!!")

    build_time = time.time() - t0

    ##### IMPORTANT! #####
    # Change index size
    
    # uncomment this for QUERY running:
    memory_usage_after = algo.get_memory_usage()
    print("Memory usage after building index: ", memory_usage_after)
    index_size = memory_usage_after - memory_usage_before 
    
    # uncomment this for GROUND TRUTH:
    # index_size = memory_usage_before

    # END: Change index size
    print("Built index in", build_time)
    print("Index size: ", index_size, flush=True)

    return build_time, index_size


# --- CHANGE: The main run function to support filters -------------------------------

# run function to check query plans in PG-Vector
'''
def run(definition: Definition, dataset_name: str, dataset_size: str, run_count: int, batch: bool) -> None:
    """Run the algorithm benchmarking.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        max_k (int): The maximum number of nearest neighbors to return.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
    """
    # Map dataset sizes to their corresponding names for paths
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), f"""\
error: query argument groups have been specified for {definition.module}.{definition.constructor}({definition.arguments}), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function"""

    for dataset_type in ["movies", "reviews"]: # hardcoded for now, can be changed later ["movies", "reviews"]
        print(f"Running on dataset type: {dataset_type}")
        X_train_ids, X_train, X_attrs, dimension = load_train_dataset(dataset_type, dataset_size)
        filter_ids, filters = load_filters(dataset_type, dataset_size)

        try:
            if hasattr(algo, "supports_prepared_queries"):
                algo.supports_prepared_queries()

            build_time, index_size = build_index(algo, X_train_ids, X_train, X_attrs, dataset_type)

            for att_idx in [0, 1]:
                if att_idx: algo.fit_idx(dataset_type)
                for fid, ff in zip(filter_ids, filters):
                    kk_values = [10]
                    X_test, distance = load_workload_dataset(dataset_type, fid, dataset_size)
                    ff = parse_filter(ff)
                    print(f"Running with filter: {ff}")
                    # if ff == ['No_filter']:
                    #    print("Skipping filter for No_filter")
                    #    continue
                    for kk in kk_values:
                        query_argument_groups = definition.query_argument_groups.copy() or [[]]  # Ensure at least one iteration
                        extra_query_arguments = [[k] for k in kk_values[:-1] if k >= kk] # except last k value
                        # append query_argument_groups with extra_query_arguments
                        query_argument_groups.extend(extra_query_arguments)
                        print("Running on query argument groups:", query_argument_groups)
                        
                        for pos, query_arguments in enumerate(query_argument_groups, 1): 
                        
                            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
                            if query_arguments:
                                algo.set_query_arguments(*query_arguments)
                                
                                for q_id in range(3):
                                    descriptor, results = run_individual_query(algo, X_train, X_test, distance, kk, run_count, batch, ff)
                                    # print("Results:", results)
                                    # raise Exception("Debugging")

                                    descriptor.update({
                                        "build_time": build_time,
                                        "index_size": index_size,
                                        "algo": definition.algorithm,
                                        "dataset": dataset_name
                                    })

                                    print(f"======= Query plan for the following search parameters:")
                                    print(f"======= Dataset type: {dataset_type}, Dataset size: {dataset_size}, M / EF Construction: {definition.arguments[1]}")
                                    print(f"======= Filter: {ff}, K: {kk}, EF Search: {query_arguments[0]}, K: {kk}, Att idx: {att_idx}")
                                    print(f"======= Results\n: {results}")
                                    # store_results(dataset_name, kk, definition, query_arguments, descriptor, results, batch, fid, dataset_size, dataset_type, att_idx)
                    

        finally:
            algo.done()
'''

# run function changed to support filters + workload modes
def run(
    definition: Definition,
    dataset_name: str,
    dataset_size: str,
    run_count: int,
    batch: bool,
    segment_size: Optional[int] = None,
    workload: str = "flex",
    corpus: str = "morevec",
    qo_sample_frac: float = DEFAULT_QO_SAMPLE_FRAC,
    max_queries: Optional[int] = None,
) -> None:
    """Run the algorithm benchmarking.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        max_k (int): The maximum number of nearest neighbors to return.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
        workload (str): ``flex`` (default), ``hard``, ``superhard``, or ``native``.
        corpus (str): ``morevec`` (default) or ``caselaw``.
    """
    if workload not in ("flex", "hard", "superhard", "native"):
        raise ValueError(f"Unknown workload: {workload}")
    if corpus not in CORPORA:
        raise ValueError(f"Unknown corpus: {corpus}")
    if corpus == "morevec" and workload == "native":
        raise ValueError("workload='native' is not supported for corpus='morevec'")
    if corpus == "caselaw":
        workload = "native"
        validate_qo_sample_frac(qo_sample_frac)

    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), f"""\
error: query argument groups have been specified for {definition.module}.{definition.constructor}({definition.arguments}), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function"""

    dataset_types = tables_for("caselaw") if corpus == "caselaw" else ["movies", "reviews"]
    for dataset_type in dataset_types:
        print(f"Running on dataset type: {dataset_type} (corpus={corpus}, workload={workload})")
        use_lazy = corpus == "caselaw" and definition.algorithm in (
            "faiss-ivf",
            "faiss-ivf-post",
            "hnsw(faiss)",
            "hnsw(faiss)-post",
        )
        X_train_ids, X_train, X_attrs, dimension = load_train_dataset(
            dataset_type, dataset_size, lazy_vectors=use_lazy
        )
        attrs_dict = build_attrs_dict(X_attrs, dataset_type)

        try:
            if hasattr(algo, "supports_prepared_queries"):
                algo.supports_prepared_queries()

            build_time, index_size = build_index(algo, X_train_ids, X_train, X_attrs, dataset_type, definition.algorithm)

            if workload == "native":
                _run_native_workload(
                    algo,
                    definition,
                    dataset_name,
                    dataset_size,
                    dataset_type,
                    run_count,
                    batch,
                    X_train,
                    attrs_dict,
                    build_time,
                    index_size,
                    qo_sample_frac=qo_sample_frac,
                    max_queries=max_queries,
                )
            elif workload in ("hard", "superhard"):
                _run_hard_workload(
                    algo, definition, dataset_name, dataset_size, dataset_type,
                    run_count, batch, workload, X_train, attrs_dict,
                    build_time, index_size,
                )
            else:
                _run_flex_workload(
                    algo, definition, dataset_name, dataset_size, dataset_type,
                    run_count, batch, X_train, X_attrs, attrs_dict,
                    build_time, index_size,
                )

        finally:
            if dataset_type == "reviews" and definition.algorithm == "milvus-ivfflat":
                algo.done(final_call=True)
            else:
                algo.done()


def _run_flex_workload(
    algo, definition, dataset_name, dataset_size, dataset_type,
    run_count, batch, X_train, X_attrs, attrs_dict, build_time, index_size,
):
    """Existing flex path: fid loop, k∈{10,20,40}, attidx results."""
    filter_ids, filters = load_filters(dataset_type, dataset_size)
    for att_idx in [0]:
        if att_idx:
            algo.fit_idx(dataset_type)
        for fid, ff in zip(filter_ids, filters):
            kk_values = [10, 20, 40]
            X_test, distance = load_workload_dataset(dataset_type, fid, dataset_size)
            ff = parse_filter(ff)
            print(f"Running with filter: {ff}")
            for kk in kk_values:
                query_argument_groups = definition.query_argument_groups.copy() or [[]]
                print(definition.algorithm)
                if definition.algorithm in ["pgvector_ivf", "faiss-ivf", "faiss-ivf-post", "milvus-ivfflat"]:
                    if dataset_type == "reviews":
                        extra_query_arguments = [[query_argument_groups[-1][-1] * 2]]
                    else:
                        extra_query_arguments = []
                else:
                    extra_query_arguments = [[k] for k in kk_values[:-1] if k >= kk]
                query_argument_groups.extend(extra_query_arguments)
                print("Running on query argument groups:", query_argument_groups)

                for pos, query_arguments in enumerate(query_argument_groups, 1):
                    print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
                    if query_arguments:
                        algo.set_query_arguments(*query_arguments)

                        if definition.algorithm in FAISS_FILTERED_ALGOS and ff != ["No_filter"]:
                            descriptor, results = run_individual_query(
                                algo, X_train, X_test, distance, kk, run_count, batch, ff, attrs_dict, True
                            )
                        else:
                            descriptor, results = run_individual_query(
                                algo, X_train, X_test, distance, kk, run_count, batch, ff
                            )

                        descriptor.update({
                            "build_time": build_time,
                            "index_size": index_size,
                            "algo": definition.algorithm,
                            "dataset": dataset_name,
                        })
                        store_results(
                            dataset_name, kk, definition, query_arguments, descriptor, results,
                            batch, fid, dataset_size, dataset_type, att_idx, workload="flex",
                        )


def _run_hard_workload(
    algo, definition, dataset_name, dataset_size, dataset_type,
    run_count, batch, workload, X_train, attrs_dict, build_time, index_size,
):
    """Hard/superhard packs: per-query filters, k=10 only, no fid loop."""
    pack = get_hard_workload_dataset(dataset_type, workload, dataset_size)
    try:
        X_test = numpy.array(pack["test"])
        raw_filters = pack["filter"][:]
        filters = [
            parse_filter(f.decode("utf-8") if isinstance(f, bytes) else str(f))
            for f in raw_filters
        ]
        distance = "angular"
        kk = 10
        print(
            f"Hard pack {workload}/{dataset_type}: {len(X_test)} queries, "
            f"{len(set(tuple(f) for f in filters))} unique parsed filters"
        )

        query_argument_groups = definition.query_argument_groups.copy() or [[]]
        if definition.algorithm in ["pgvector_ivf", "faiss-ivf", "faiss-ivf-post", "milvus-ivfflat"]:
            if dataset_type == "reviews" and query_argument_groups and query_argument_groups[-1]:
                extra_query_arguments = [[query_argument_groups[-1][-1] * 2]]
            else:
                extra_query_arguments = []
            query_argument_groups.extend(extra_query_arguments)
        print("Running on query argument groups:", query_argument_groups)

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
            if query_arguments:
                algo.set_query_arguments(*query_arguments)

            if definition.algorithm in FAISS_FILTERED_ALGOS:
                descriptor, results = run_individual_query(
                    algo, X_train, X_test, distance, kk, run_count, batch,
                    ["No_filter"], attrs_dict, True, per_query_filters=filters,
                )
            else:
                descriptor, results = run_individual_query(
                    algo, X_train, X_test, distance, kk, run_count, batch,
                    ["No_filter"], per_query_filters=filters,
                )

            descriptor.update({
                "build_time": build_time,
                "index_size": index_size,
                "algo": definition.algorithm,
                "dataset": dataset_name,
                "workload": workload,
                "data_table": dataset_type,
            })
            store_results(
                dataset_name, kk, definition, query_arguments, descriptor, results,
                batch, 0, dataset_size, dataset_type, 0, workload=workload,
            )
    finally:
        pack.close()


def _run_native_workload(
    algo,
    definition,
    dataset_name,
    dataset_size,
    dataset_type,
    run_count,
    batch,
    X_train,
    attrs_dict,
    build_time,
    index_size,
    qo_sample_frac: float = DEFAULT_QO_SAMPLE_FRAC,
    max_queries: Optional[int] = None,
):
    """CaseLaw native workload: per-query predicates from manifest order, k=10."""
    manifest_fn = sample_manifest_path("caselaw")
    with open(manifest_fn, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)
    records = list(manifest["queries"])
    if max_queries is not None:
        if int(max_queries) < 1:
            raise ValueError(f"max_queries must be >= 1, got {max_queries}")
        records = records[: int(max_queries)]
    q_indices = numpy.asarray([int(r["query_index"]) for r in records], dtype=numpy.int64)

    pack = get_native_workload_dataset(dataset_type)
    try:
        all_test = numpy.asarray(pack["test"][:], dtype=numpy.float32)
        if hasattr(pack["filter"], "asstr"):
            all_filters = pack["filter"].asstr()[:]
        else:
            all_filters = [
                f.decode("utf-8") if isinstance(f, (bytes, numpy.bytes_)) else str(f)
                for f in pack["filter"][:]
            ]
        _gt_ids = numpy.asarray(pack["gt_ids"][:], dtype=numpy.int32)[q_indices]
        _nn_dists = numpy.asarray(pack["nn_dists"][:], dtype=numpy.float32)[q_indices]
        _sels = numpy.asarray([float(r["selectivity"]) for r in records], dtype=numpy.float64)
        _nres = numpy.asarray([int(r["nres"]) for r in records], dtype=numpy.int32)
        X_test = all_test[q_indices]
        filters = [parse_predicate(str(all_filters[int(qi)])) for qi in q_indices]
        distance = str(pack.attrs.get("metric", "euclidean"))
        kk = 10
        print(
            f"Native pack {dataset_type}: {len(X_test)} queries "
            f"(n_population={manifest['n_population']}, n_train={manifest['n_train']}, "
            f"n_test={manifest['n_test']}, metric={distance})"
        )

        query_argument_groups = definition.query_argument_groups.copy() or [[]]
        print("Running on query argument groups:", query_argument_groups)

        if (
            not batch
            and run_count == 1
            and len(X_test) >= 1
            and definition.algorithm in ("faiss-flat", "faiss-ivf", "faiss-ivf-post", "hnsw(faiss)", "hnsw(faiss)-post")
        ):
            print(
                f"Precomputing {len(query_argument_groups)} hyperparameter groups in one pass for {definition.algorithm}...",
                flush=True,
            )
            mhp = _precompute_native_multi_hp(
                algo,
                definition.algorithm,
                X_train,
                X_test,
                distance,
                kk,
                attrs_dict,
                filters,
                query_argument_groups,
                gt_ids=_gt_ids,
                selectivities=_sels,
                nres=_nres,
                nn_dists=_nn_dists,
            )
            algo._precomputed_hp_queue = [mhp[i] for i in range(len(query_argument_groups))]

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
            if query_arguments:
                algo.set_query_arguments(*query_arguments)

            if definition.algorithm in FAISS_FILTERED_ALGOS:
                descriptor, results = run_individual_query(
                    algo,
                    X_train,
                    X_test,
                    distance,
                    kk,
                    run_count,
                    batch,
                    ["No_filter"],
                    attrs_dict,
                    True,
                    per_query_filters=filters,
                )
            else:
                descriptor, results = run_individual_query(
                    algo,
                    X_train,
                    X_test,
                    distance,
                    kk,
                    run_count,
                    batch,
                    ["No_filter"],
                    per_query_filters=filters,
                )

            descriptor.update(
                {
                    "build_time": build_time,
                    "index_size": index_size,
                    "algo": definition.algorithm,
                    "dataset": dataset_name,
                    "workload": "native",
                    "data_table": dataset_type,
                    "corpus": "caselaw",
                    "metric": distance,
                    "distance": distance,
                    "qo_sample_frac": float(manifest.get("frac", qo_sample_frac)),
                    "n_train": int(manifest["n_train"]),
                    "n_test": int(manifest["n_test"]),
                    "n_population": int(manifest["n_population"]),
                }
            )
            if max_queries is not None:
                descriptor["max_queries"] = int(max_queries)

            store_results(
                dataset_name,
                kk,
                definition,
                query_arguments,
                descriptor,
                results,
                batch,
                0,
                dataset_size,
                dataset_type,
                0,
                workload="native",
            )
    finally:
        pack.close()


# --------------------------------------------------------------

def run_from_cmdline():
    """Calls the function `run` using arguments from the command line. See `ArgumentParser` for 
    arguments, all run it with `--help`.
    """
    parser = argparse.ArgumentParser(
        """

            NOTICE: You probably want to run.py rather than this script.

"""
    )
    parser.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to benchmark on.", required=True)
    parser.add_argument("--algorithm", help="Name of algorithm for saving the results.", required=True)
    parser.add_argument(
        "--module", help='Python module containing algorithm. E.g. "ann_benchmarks.algorithms.annoy"', required=True
    )
    parser.add_argument("--constructor", help='Constructer to load from modulel. E.g. "Annoy"', required=True)
    #parser.add_argument(
    #    "--max_k", help="K: Max number of nearest neighbours for the algorithm to return.", required=True, type=int
    #)
    parser.add_argument("--dataset_size", choices=["large", "medium", "small"], help="Choose dataset size from: large, medium, small", required=True)
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorihm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch",
        help='If flag included, algorithms will be run in batch mode, rather than "individual query" mode.',
        action="store_true",
    )
    parser.add_argument("--segment_size", type=int, default=None,
                        help="Milvus segment size in MB for ablation study (extracted from docker_tag when running from host)")
    parser.add_argument(
        "--workload",
        choices=["flex", "hard", "superhard", "native"],
        default="flex",
        help="Workload mode: flex (default fid loop), hard, superhard, or native packs.",
    )
    parser.add_argument(
        "--corpus",
        choices=CORPORA,
        default="morevec",
        help="Corpus to run: morevec (default) or caselaw.",
    )
    parser.add_argument(
        "--qo-sample-frac",
        type=float,
        default=DEFAULT_QO_SAMPLE_FRAC,
        help="Training sample fraction for CaseLaw manifest.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional prefix truncation on manifest queries (smoke only).",
    )
    # New argument to parser
    parser.add_argument(
        "--filter",
        help='If flag included, algorithms will look for dataset with given filter in the name.',
        type=str,
        default=None,
    )
    
    parser.add_argument("build", help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]')
    parser.add_argument("queries", help="JSON of arguments to pass to the queries. E.g. [100]", nargs="*", default=[])
    args = parser.parse_args()

    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False,
    )
    run(
        definition,
        args.dataset,
        args.dataset_size,
        args.runs,
        args.batch,
        args.segment_size,
        workload=args.workload,
        corpus=args.corpus,
        qo_sample_frac=args.qo_sample_frac,
        max_queries=args.max_queries,
    )


def run_docker(
    definition: Definition,
    dataset: str,
    dataset_size: str,
    runs: int,
    timeout: int,
    batch: bool,
    cpu_limit: str,
    mem_limit: Optional[int] = None,
    workload: str = "flex",
    corpus: str = "morevec",
    qo_sample_frac: float = DEFAULT_QO_SAMPLE_FRAC,
    max_queries: Optional[int] = None,
) -> None:
    """Runs `run_from_cmdline` within a Docker container with specified parameters and logs the output.

    See `run_from_cmdline` for details on the args.
    """
    # Extract segment size from docker_tag (e.g., "ann-benchmarks-milvus-seg16384" -> 16384)
    segment_size = 1024  # default
    if definition.docker_tag and "-seg" in definition.docker_tag:
        try:
            segment_size = int(definition.docker_tag.split("-seg")[-1])
        except ValueError:
            pass

    cmd = [
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
        "--dataset_size",
        dataset_size,
        "--segment_size",
        str(segment_size),
        "--workload",
        workload,
        "--corpus",
        corpus,
        "--qo-sample-frac",
        str(qo_sample_frac),
    ]
    if max_queries is not None:
        cmd += ["--max-queries", str(max_queries)]
    if batch:
        cmd += ["--batch"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available
    print("Running with CPU limit:", cpu_limit, "and memory limit:", mem_limit)
    # Pass host project dir so the container can rewrite docker-compose.yml
    # volume paths to be accessible by snap Docker (which can't see /tmp)
    host_project_dir = os.path.abspath(".")
    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath("/var/run/docker.sock"): {"bind": "/var/run/docker.sock", "mode": "rw"},
            os.path.abspath("ann_benchmarks"): {"bind": "/home/app/ann_benchmarks", "mode": "ro"},
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
        },
        environment={
            "HOST_PROJECT_DIR": host_project_dir,
        },
        network_mode="host",
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        # Size /dev/shm to the container's memory budget. PostgreSQL parallel
        # index builds (e.g. pgvector HNSW with max_parallel_maintenance_workers
        # > 0) place the shared graph in a POSIX dynamic-shared-memory segment
        # under /dev/shm, sized up to maintenance_work_mem. Docker's default
        # /dev/shm is only 64 MB, so large builds fail with "could not resize
        # shared memory segment ... No space left on device". Matching shm_size
        # to mem_limit lets the build use the same memory budget on every host.
        shm_size=mem_limit,
        detach=True,
    )
    logger = logging.getLogger(f"annb.{container.short_id}")

    # NEW: check installed packages in the container
    # pip_list_result = container.exec_run("pip list")
    # logger.info(f"Installed packages in container {container.short_id}:\n{pip_list_result.output.decode()}")
    
    logger.info(
        "Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s"
        % (container.short_id, cpu_limit, mem_limit, timeout, cmd)
    )

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        return_value = container.wait(timeout=timeout)
        _handle_container_return_value(return_value, container, logger)
    except Exception as e:
        logger.error("Container.wait for container %s failed with exception", container.short_id)
        logger.error(str(e))
    finally:
        logger.info("Removing container")
        container.remove(force=True)


def _handle_container_return_value(
    return_value: Union[Dict[str, Union[int, str]], int],
    container: docker.models.containers.Container,
    logger: logging.Logger
) -> None:
    """Handles the return value of a Docker container and outputs error and stdout messages (with colour).

    Args:
        return_value (Union[Dict[str, Union[int, str]], int]): The return value of the container.
        container (docker.models.containers.Container): The Docker container.
        logger (logging.Logger): The logger instance.
    """

    base_msg = f"Child process for container {container.short_id} "
    msg = base_msg + "returned exit code {}"

    if isinstance(return_value, dict):  # The return value from container.wait changes from int to dict in docker 3.0.0
        error_msg = return_value.get("Error", "")
        exit_code = return_value["StatusCode"]
        msg = msg.format(f"{exit_code} with message {error_msg}")
    else:
        exit_code = return_value
        msg = msg.format(exit_code)

    if exit_code not in [0, None]:
        for line in container.logs(stream=True):
            logger.error(colors.color(line.decode(), fg="red"))
        logger.error(msg)
    else:
        logger.info(msg)
