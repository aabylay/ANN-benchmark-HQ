import numpy


DEFAULT_GAMMA = 1.0
MAX_SEARCH_K = 1000


def filter_mask_from_fvalue(X_attr, fvalue):
    """Build a boolean mask for the single-attribute filter used in benchmarks."""
    op = fvalue[1]
    threshold = float(fvalue[2])
    if op == ">=":
        return X_attr >= threshold
    if op == "<=":
        return X_attr <= threshold
    if op == ">":
        return X_attr > threshold
    if op == "<":
        return X_attr < threshold
    if op == "=":
        return X_attr == threshold
    if op == "!=":
        return X_attr != threshold
    raise ValueError(f"Unsupported filter operator: {op}")


def compute_search_k(target_k, selectivity, gamma=DEFAULT_GAMMA, max_k=MAX_SEARCH_K):
    """Enriched candidate pool size for post-filtering: min(max_k, target_k / sel * gamma)."""
    if selectivity <= 0:
        return max_k
    k = int(numpy.ceil(target_k / selectivity * gamma))
    return min(max_k, max(target_k, k))


def apply_post_filter(indices, filter_mask, target_k):
    """Keep the first target_k candidates (in search order) that pass filter_mask."""
    kept = []
    for idx in indices:
        if idx == -1:
            continue
        if filter_mask[int(idx)]:
            kept.append(int(idx))
            if len(kept) >= target_k:
                break
    return numpy.array(kept, dtype=numpy.int64)
