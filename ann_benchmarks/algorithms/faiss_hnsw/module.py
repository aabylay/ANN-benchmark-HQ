import os
import faiss
from faiss import swig_ptr
import numpy as np

from ..faiss.module import Faiss, _attrs_for_query, _bitmap_from_fvalue
from ..faiss.postfilter import apply_post_filter, compute_search_k, filter_mask_from_fvalue
from ...attrs import (
    _is_clause_list,
    coerce_attrs,
    filter_mask_from_attrs,
    postfilter_candidates_from_attrs,
    selectivity_from_attrs,
)


class FaissHNSW(Faiss):
    def __init__(self, metric, method_param):
        self._metric = metric
        self.method_param = method_param
        self._query_dist_comps = []
        self._last_dist_comps = 0

    def fit(self, X_ids, X, X_att, dataset_type):
        print("Index params:", self.method_param["M"], self.method_param["efConstruction"])
        if self._metric == "angular":
            self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"], faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"])
        self.index.hnsw.efConstruction = self.method_param["efConstruction"]
        self.index.verbose = True
        self._attrs = coerce_attrs(X_att, dataset_type)
        self._dataset_type = dataset_type

        if self._metric == "angular":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        faiss.omp_set_num_threads(48)
        read_cache_path = None
        write_cache_path = None
        if dataset_type == "cases":
            data_cache_dir = os.path.join("data", "datasets", "CaseLaw_7M", "index_cache")
            res_cache_dir = os.path.join("results", "CaseLaw_7M_native_cases", ".index_cache")
            os.makedirs(res_cache_dir, exist_ok=True)
            m_val = int(self.method_param["M"])
            efc_val = int(self.method_param["efConstruction"])
            fname = f"hnsw_{self._metric}_n{X.shape[0]}_d{X.shape[1]}_M{m_val}_efc{efc_val}.faiss"
            p1 = os.path.join(data_cache_dir, fname)
            p2 = os.path.join(res_cache_dir, fname)
            read_cache_path = p1 if os.path.isfile(p1) else (p2 if os.path.isfile(p2) else None)
            write_cache_path = p2
        if read_cache_path:
            print(f"FaissHNSW: loading cached index from {read_cache_path}", flush=True)
            self.index = faiss.read_index(read_cache_path)
        else:
            X = np.asarray(X, dtype=np.float32)
            self.index.add(X)
            if write_cache_path:
                tmp_path = f"{write_cache_path}.{os.getpid()}.part"
                faiss.write_index(self.index, tmp_path)
                os.replace(tmp_path, write_cache_path)
                print(f"FaissHNSW: saved cached index to {write_cache_path}", flush=True)

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self._query_dist_comps = []
        self._last_dist_comps = 0
        self.index.hnsw.efSearch = ef
        self.index.hnsw.check_relative_distance = True

    def _search(self, v, n, params=None):
        """HNSW search that records the number of distance computations."""
        prev = int(faiss.cvar.hnsw_stats.ndis)
        if params is None:
            D, I = self.index.search(v, n)
        else:
            D, I = self.index.search(v, n, params=params)
        ndis = int(faiss.cvar.hnsw_stats.ndis) - prev
        self._last_dist_comps = ndis
        self._query_dist_comps.append(ndis)
        return D, I

    def query(self, v, n, fvalue=["No_filter"], X_attr=None):
        v = np.expand_dims(v, axis=0).astype(np.float32)
        if fvalue == ["No_filter"]:
            _, I = self._search(v, n)
        else:
            search_params = faiss.SearchParametersHNSW()
            # Keep the bitmap alive until the search completes: IDSelectorBitmap
            # stores a raw pointer to the numpy buffer, so passing a temporary
            # is a use-after-free that corrupts the filter mask.
            bitmap = _bitmap_from_fvalue(self, fvalue, X_attr)
            sel = faiss.IDSelectorBitmap(bitmap)
            search_params.efSearch = self.index.hnsw.efSearch
            search_params.sel = sel
            search_params.check_relative_distance = self.index.hnsw.check_relative_distance
            _, I = self._search(v, n, params=search_params)
        return I[0]

    def get_additional(self):
        return {"dist_comps": faiss.cvar.hnsw_stats.ndis}

    def get_query_dist_comps(self):
        return list(self._query_dist_comps)

    def __str__(self):
        return "faiss (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)

    def freeIndex(self):
        del self.p


class FaissHNSWPostFilter(FaissHNSW):
    """HNSW with post-filtering: over-fetch candidates, then filter in numpy."""

    def query(self, v, n, fvalue=["No_filter"], X_attr=None):
        if fvalue == ["No_filter"]:
            return super().query(v, n, fvalue, X_attr)

        attrs = _attrs_for_query(self, X_attr)
        if "_single" in attrs and len(attrs) == 1:
            filter_mask = filter_mask_from_fvalue(attrs["_single"], fvalue)
            selectivity = float(filter_mask.mean())
            search_k = compute_search_k(n, selectivity)
            search_params = faiss.SearchParametersHNSW()
            search_params.efSearch = max(self.index.hnsw.efSearch, search_k)
            search_params.check_relative_distance = self.index.hnsw.check_relative_distance
            v = np.expand_dims(v, axis=0).astype(np.float32)
            _, I = self._search(v, search_k, params=search_params)
            return apply_post_filter(I[0], filter_mask, n)
        if _is_clause_list(fvalue):
            selectivity = selectivity_from_attrs(attrs, fvalue)
            search_k = compute_search_k(n, selectivity)
            search_params = faiss.SearchParametersHNSW()
            search_params.efSearch = max(self.index.hnsw.efSearch, search_k)
            search_params.check_relative_distance = self.index.hnsw.check_relative_distance
            v = np.expand_dims(v, axis=0).astype(np.float32)
            _, I = self._search(v, search_k, params=search_params)
            return postfilter_candidates_from_attrs(attrs, fvalue, I[0], n)
        filter_mask = filter_mask_from_attrs(attrs, fvalue)
        selectivity = float(filter_mask.mean())
        search_k = compute_search_k(n, selectivity)

        search_params = faiss.SearchParametersHNSW()
        search_params.efSearch = max(self.index.hnsw.efSearch, search_k)
        search_params.check_relative_distance = self.index.hnsw.check_relative_distance

        v = np.expand_dims(v, axis=0).astype(np.float32)
        _, I = self._search(v, search_k, params=search_params)
        return apply_post_filter(I[0], filter_mask, n)

    def __str__(self):
        return "faiss-hnsw-post (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)
