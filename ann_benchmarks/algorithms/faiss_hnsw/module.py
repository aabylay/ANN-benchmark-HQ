import faiss
from faiss import swig_ptr
import numpy as np

from ..faiss.module import Faiss
from ..faiss.postfilter import apply_post_filter, compute_search_k, filter_mask_from_fvalue


class FaissHNSW(Faiss):
    def __init__(self, metric, method_param):
        self._metric = metric
        self.method_param = method_param

    def fit(self, X_ids, X, X_att, dataset_type):
        print("Index params:", self.method_param["M"], self.method_param["efConstruction"])
        if self._metric == "angular":
            self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"], faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"])
        self.index.hnsw.efConstruction = self.method_param["efConstruction"]
        self.index.verbose = True

        if self._metric == "angular":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        X_att = X_att.astype(np.float32)
        # print("X_att shape:", X_att.shape)
        # print("X_att:", X_att[:10])
            
        self.index.add(X)
        #self.index.add_att(X.shape[0], swig_ptr(X_att)) # new line
        faiss.omp_set_num_threads(48)

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self.index.hnsw.efSearch = ef
        self.index.hnsw.check_relative_distance = True

    def query(self, v, n, fvalue = ["No_filter"], X_attr = None):
        #if self._metric == "angular":
        #    v /= numpy.linalg.norm(v)
        # print("", np.expand_dims(v, axis=0).astype(np.float32).shape)
        # v_test = np.expand_dims(v, axis=0).astype(np.float32)
        # print(v_test.shape)
        # raise Exception("Debugging")
        
        if fvalue == ["No_filter"]:
            D, I = self.index.search((np.expand_dims(v, axis=0).astype(np.float32)), n)
        else:
            """
            D = np.empty((1, n), dtype=np.float32)
            I = np.empty((1, n), dtype=np.int64)
            v = np.expand_dims(v, axis=0).astype('float32')
            #print("Filter value:", float(fvalue[2]))
            self.index.filtered_search(1, swig_ptr(v), n, swig_ptr(D), swig_ptr(I), float(fvalue[2]))
            """
            # print("Filter value:", fvalue)
            search_params = faiss.SearchParametersHNSW()
            bitmap_bool = X_attr >= float(fvalue[2])
            bitmap = np.packbits(bitmap_bool, bitorder='little')
            bitmap = np.ascontiguousarray(bitmap, dtype=np.uint8)
            sel = faiss.IDSelectorBitmap(bitmap)
            search_params.efSearch = self.index.hnsw.efSearch
            search_params.sel = sel
            search_params.check_relative_distance = self.index.hnsw.check_relative_distance
            D, I = self.index.search(np.expand_dims(v, axis=0).astype(np.float32), n, params=search_params)
        #print("I:", I)
        return I[0]
    
    def get_additional(self):
        return {"dist_comps": faiss.cvar.hnsw_stats.ndis}

    def __str__(self):
        return "faiss (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)

    def freeIndex(self):
        del self.p


class FaissHNSWPostFilter(FaissHNSW):
    """HNSW with post-filtering: over-fetch candidates, then filter in numpy."""

    def query(self, v, n, fvalue=["No_filter"], X_attr=None):
        if fvalue == ["No_filter"]:
            return super().query(v, n, fvalue, X_attr)

        filter_mask = filter_mask_from_fvalue(X_attr, fvalue)
        selectivity = float(filter_mask.mean())
        search_k = compute_search_k(n, selectivity)

        search_params = faiss.SearchParametersHNSW()
        search_params.efSearch = max(self.index.hnsw.efSearch, search_k)
        search_params.check_relative_distance = self.index.hnsw.check_relative_distance

        v = np.expand_dims(v, axis=0).astype(np.float32)
        _, I = self.index.search(v, search_k, params=search_params)
        return apply_post_filter(I[0], filter_mask, n)

    def __str__(self):
        return "faiss-hnsw-post (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)
