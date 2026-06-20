import sys, h5py, numpy as np, pandas as pd

LARGE = "data/datasets/MoRe_large"

def read_filters(f):
    with h5py.File(f, "r") as h:
        filts = [x.decode() if isinstance(x, bytes) else x for x in h["filters"][()]]
        sels = list(h["selectivities"][()])
    return dict(zip(filts, sels))

def thr(filters_dict, col):
    out = {}
    for k in filters_dict:
        if k.startswith(col + " >="):
            out[float(k.split(">=")[1])] = filters_dict[k]
    return out

m_hdf5 = thr(read_filters(f"{LARGE}/filters/movies_filters_0.hdf5"), "avg_rating")
r_hdf5 = thr(read_filters(f"{LARGE}/filters/reviews_filters_0.hdf5"), "total_votes")

# Actual large-dataset columns
with h5py.File(f"{LARGE}/datasets/movies_dataset_0.hdf5", "r") as h:
    avgrating = h["train_avgrating"][()]
print(f"movies rows = {len(avgrating)}")
with h5py.File(f"{LARGE}/datasets/reviews_dataset_0.hdf5", "r") as h:
    # find the totalvotes-like column
    cols = list(h.keys())
    print("reviews cols:", cols)
    tv_key = [c for c in cols if "vote" in c.lower() and "total" in c.lower()]
    tv_key = tv_key[0] if tv_key else [c for c in cols if "vote" in c.lower()][0]
    totalvotes = h[tv_key][()]
print(f"reviews rows = {len(totalvotes)} (col={tv_key})")

def check(name, col_vals, hdf5_thr):
    print(f"\n=== {name}: recompute selectivity on ACTUAL large data ===")
    n = len(col_vals)
    print(f"{'threshold':>12} {'sel_in_hdf5':>14} {'sel_recomputed':>16} {'match?':>8}")
    for t in sorted(hdf5_thr):
        recomputed = np.mean(col_vals >= t)
        ok = abs(recomputed - hdf5_thr[t]) < 1e-4
        print(f"{t:>12} {hdf5_thr[t]:>14.6f} {recomputed:>16.6f} {'OK' if ok else 'NO':>8}")

check("MOVIES avg_rating", avgrating, m_hdf5)
check("REVIEWS total_votes", totalvotes, r_hdf5)

# Now show what the CSV (gls corr) thresholds would give on the large data
csv = pd.read_csv(f"{LARGE}/stats/filter_stats_0.csv")
def csv_thr(qt, col):
    sub = csv[csv["query_type"] == qt]
    out = {}
    for f in sub["filter"].unique():
        if isinstance(f, str) and f.startswith(col + " >="):
            out[float(f.split(">=")[1])] = sub[sub["filter"] == f]["selectivity"].iloc[0]
    return out

print("\n=== GLS-corr CSV thresholds re-evaluated on the LARGE data ===")
for name, vals, ct in [("MOVIES avg_rating", avgrating, csv_thr("flex_movies_sim", "avg_rating")),
                       ("REVIEWS total_votes", totalvotes, csv_thr("flex_reviews_sim", "total_votes"))]:
    print(f"-- {name} --")
    for t in sorted(ct):
        recomputed = np.mean(vals >= t)
        print(f"   thr={t:>10}  sel_in_csv={ct[t]:.6f}  sel_on_large={recomputed:.6f}  in_large_filters={'YES' if t in (m_hdf5 if 'MOVIES' in name else r_hdf5) else 'NO'}")
sys.stdout.flush()
