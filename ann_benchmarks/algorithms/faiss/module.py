import sys

sys.path.append("install/lib-faiss")  # noqa
import faiss
import numpy
import sklearn.preprocessing

from faiss import swig_ptr
from ..base.module import BaseANN
from ...attrs import coerce_attrs, filter_mask_from_attrs
from .postfilter import apply_post_filter, compute_search_k, filter_mask_from_fvalue


def _attrs_for_query(algo, X_attr):
    """Prefer attrs stored at fit(); fall back to query-time X_attr."""
    stored = getattr(algo, "_attrs", None)
    if stored is not None:
        return stored
    if isinstance(X_attr, dict):
        return X_attr
    if X_attr is None:
        raise ValueError("FAISS filtered query requires attrs (fit dict or X_attr)")
    return {"_single": numpy.asarray(X_attr, dtype=numpy.float32)}


def _bitmap_from_fvalue(algo, fvalue, X_attr):
    attrs = _attrs_for_query(algo, X_attr)
    if "_single" in attrs and len(attrs) == 1:
        bitmap_bool = filter_mask_from_fvalue(attrs["_single"], fvalue)
    else:
        bitmap_bool = filter_mask_from_attrs(attrs, fvalue)
    bitmap = numpy.packbits(bitmap_bool, bitorder="little")
    return numpy.ascontiguousarray(bitmap, dtype=numpy.uint8)


class Faiss(BaseANN):
    def query(self, v, n, fvalue = ["No_filter"], X_attr = None):
        if self._metric == "angular":
            v /= numpy.linalg.norm(v)
        if fvalue == ["No_filter"]:
            pass
            D, I = self.index.search(numpy.expand_dims(v, axis=0).astype(numpy.float32), n)
        else:
            search_params = faiss.SearchParametersIVF()
            bitmap = _bitmap_from_fvalue(self, fvalue, X_attr)
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
        # Actual number of clusters used to build the index. When the configured
        # value is <= 0 it is auto-derived per table as round(sqrt(|D|)) in fit().
        self._nlist = None

    def fit(self, X_ids, X, X_att, dataset_type): # to do
        faiss.omp_set_num_threads(48)
        print("Index params:", self._n_list)
        d = int(X.shape[1])  # Cast to native int
        nlist = int(self._n_list["clusters"])  # Cast to native int (handles any upstream float)

        # Fixed construction param: clusters ~ sqrt(|D|), computed per table.
        # A configured value of 0 (or negative) requests this auto behaviour.
        if nlist <= 0:
            nlist = max(1, int(round(numpy.sqrt(X.shape[0]))))
        self._nlist = nlist
        print(f"FaissIVF: building index with nlist={nlist} for n={X.shape[0]} ({dataset_type})")
        self._attrs = coerce_attrs(X_att, dataset_type)
        self._dataset_type = dataset_type

        self.quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(self.quantizer, d, nlist)

        if self._metric == "angular":
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        self.index.train(X)
        self.index.add(X)
        

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self._nlist}  # noqa

    def __str__(self):
        return "FaissIVF(n_list=%d, n_probe=%d)" % (self._nlist, self._n_probe)


class FaissIVFPostFilter(FaissIVF):
    """IVFFlat with post-filtering: over-fetch candidates, then filter in numpy."""

    def query(self, v, n, fvalue=["No_filter"], X_attr=None):
        if self._metric == "angular":
            v /= numpy.linalg.norm(v)
        v = numpy.expand_dims(v, axis=0).astype(numpy.float32)

        if fvalue == ["No_filter"]:
            _, I = self.index.search(v, n)
            return I[0]

        attrs = _attrs_for_query(self, X_attr)
        if "_single" in attrs and len(attrs) == 1:
            filter_mask = filter_mask_from_fvalue(attrs["_single"], fvalue)
        else:
            filter_mask = filter_mask_from_attrs(attrs, fvalue)
        selectivity = float(filter_mask.mean())
        search_k = compute_search_k(n, selectivity)
        _, I = self.index.search(v, search_k)
        return apply_post_filter(I[0], filter_mask, n)

    def __str__(self):
        return "FaissIVF-post(n_list=%d, n_probe=%d)" % (self._nlist, self._n_probe)


class FaissFlat(Faiss):
    """Exact brute-force search using a flat (IndexFlatIP) index, CPU based.

    Filtered queries use the same bitset pre-filtering approach as the other
    FAISS plans (a faiss.IDSelectorBitmap built from ``X_attr >= threshold``),
    so the result is the exact filtered top-k (recall == 1.0). There are no
    construction or search parameters to sweep.
    """

    def __init__(self, metric):
        self._metric = metric
        self.index = None

    def fit(self, X_ids, X, X_att, dataset_type):
        faiss.omp_set_num_threads(48)
        d = int(X.shape[1])
        self._attrs = coerce_attrs(X_att, dataset_type)
        self._dataset_type = dataset_type
        if self._metric == "angular":
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")
            self.index = faiss.IndexFlatIP(d)
        else:
            self.index = faiss.IndexFlatL2(d)
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        self.index.add(X)
        print(f"FaissFlat: built IndexFlat{'IP' if self._metric == 'angular' else 'L2'} for n={X.shape[0]} ({dataset_type})")

    def set_query_arguments(self, placeholder=0):
        # Brute-force search has no tunable search parameters; accept and ignore
        # the placeholder argument so the runner's query-argument loop works.
        self._placeholder = placeholder

    def query(self, v, n, fvalue=["No_filter"], X_attr=None):
        if self._metric == "angular":
            v = v / numpy.linalg.norm(v)
        v = numpy.expand_dims(v, axis=0).astype(numpy.float32)
        if fvalue == ["No_filter"]:
            D, I = self.index.search(v, n)
        else:
            search_params = faiss.SearchParameters()
            # Keep the bitmap alive until the search completes: IDSelectorBitmap
            # stores a raw pointer to the numpy buffer, so passing a temporary
            # is a use-after-free that corrupts the filter mask.
            bitmap = _bitmap_from_fvalue(self, fvalue, X_attr)
            search_params.sel = faiss.IDSelectorBitmap(bitmap)
            D, I = self.index.search(v, n, params=search_params)
        return I[0]

    def get_additional(self):
        return {"dist_comps": 0}

    def __str__(self):
        return "FaissFlat(metric=%s)" % self._metric


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