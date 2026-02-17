import sys

sys.path.append("install/lib-faiss")  # noqa
import faiss
import numpy
import sklearn.preprocessing

from faiss import swig_ptr
from ..base.module import BaseANN


class Faiss(BaseANN):
    def query(self, v, n, fvalue = ["No_filter"], X_attr = None):
        if self._metric == "angular":
            v /= numpy.linalg.norm(v)
        if fvalue == ["No_filter"]:
            pass
            D, I = self.index.search(numpy.expand_dims(v, axis=0).astype(numpy.float32), n)
        else:
            """
            D = numpy.empty((1, n), dtype=numpy.float32)
            I = numpy.empty((1, n), dtype=numpy.int64)
            v = numpy.expand_dims(v, axis=0).astype('float32')
            self.index.filtered_search(1, swig_ptr(v), n, swig_ptr(D), swig_ptr(I), float(fvalue[2]))
            """
            # print("Attr value:", X_attr)
            search_params = faiss.SearchParametersIVF()
            bitmap_bool = X_attr >= float(fvalue[2])
            bitmap = numpy.packbits(bitmap_bool, bitorder='little')
            bitmap = numpy.ascontiguousarray(bitmap, dtype=numpy.uint8)
            sel = faiss.IDSelectorBitmap(bitmap)
            search_params.nprobe = self.index.nprobe
            search_params.sel = sel
            D, I = self.index.search(numpy.expand_dims(v, axis=0).astype(numpy.float32), n, params=search_params)
        return I[0]

    def batch_query(self, X, n):
        if self._metric == "angular":
            X /= numpy.linalg.norm(X)
        self.res = self.index.search(X.astype(numpy.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res


class FaissLSH(Faiss):
    def __init__(self, metric, n_bits):
        self._n_bits = n_bits
        self.index = None
        self._metric = metric
        self.name = "FaissLSH(n_bits={})".format(self._n_bits)

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        f = X.shape[1]
        self.index = faiss.IndexLSH(f, self._n_bits)
        self.index.train(X)
        self.index.add(X)


class FaissIVF(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def fit(self, X_ids, X, X_att, dataset_type): # to do
        faiss.omp_set_num_threads(48)
        print("Index params:", self._n_list)
        d = int(X.shape[1])  # Cast to native int
        nlist = int(self._n_list["clusters"])  # Cast to native int (handles any upstream float)
        
        self.quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(self.quantizer, d, nlist)

        if self._metric == "angular":
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        # X_att = X_att.astype(numpy.float32)
        
        self.index.train(X)
        self.index.add(X)
        # self.index.add_att(X.shape[0], swig_ptr(X_att)) # new line
        # self.index = index
        

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self._n_list["clusters"]}  # noqa

    def __str__(self):
        return "FaissIVF(n_list=%d, n_probe=%d)" % (self._n_list["clusters"], self._n_probe)


class FaissIVFPQfs(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        if self._metric == "angular":
            faiss.normalize_L2(X)

        d = X.shape[1]
        faiss_metric = faiss.METRIC_INNER_PRODUCT if self._metric == "angular" else faiss.METRIC_L2
        factory_string = f"IVF{self._n_list},PQ{d//2}x4fs"
        index = faiss.index_factory(d, factory_string, faiss_metric)
        index.train(X)
        index.add(X)
        index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(X))
        self.base_index = index
        self.refine_index = index_refine

    def set_query_arguments(self, n_probe, k_reorder):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self._k_reorder = k_reorder
        self.base_index.nprobe = self._n_probe
        self.refine_index.k_factor = self._k_reorder
        if self._k_reorder == 0:
            self.index = self.base_index
        else:
            self.index = self.refine_index

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self._n_list}  # noqa

    def __str__(self):
        return "FaissIVFPQfs(n_list=%d, n_probe=%d, k_reorder=%d)" % (self._n_list, self._n_probe, self._k_reorder)