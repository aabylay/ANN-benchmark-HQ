#!/usr/bin/env python3
"""Per-query runtime / recall diagnostics at fixed search hyperparameters.

For each ANN index (HNSW / IVF, FAISS and pgvector), emits figures with
one row per selected search parameter and four columns:

  1. metric distribution (histogram)
  2. metric vs filter selectivity (σ)
  3. metric vs Post_Hardness
  4. metric vs GLS correlation (ρ)

Two metric families are written side by side, plus distance computations
when the CSV has a ``dist_comps`` column (FAISS HNSW):

  - ``runtime_panels_*.png`` / ``runtime_cdf_*.png``  (runtime in ms, log scale)
  - ``recall_panels_*.png``  / ``recall_cdf_*.png``   (recall in [0,1], linear)
  - ``dist_comps_panels_*.png`` / ``dist_comps_cdf_*.png``  (distance comps, log)

In addition to the mid-range HP grids, the script writes:

  - one-row ``*_panels_*_max_*`` at the largest swept ``ef_search`` / ``n_probes``
  - one-row brute-force panels (``faiss-flat``, ``pgvector_bf``)
  - 3-row ``*_panels_max_vs_bf_{faiss,faiss_post,pgvector}_*`` comparing those three

Reads the QO per-query CSV produced by ``query_optimizer_hard_analysis.py``
(or the flex analysis). Duplicate runs for the same (query, algo, hp) are
averaged.

Example::

    python analysis/query_runtime_plots.py \\
      --results-csv analysis/plots/query_optimizer_hard_movies/all_query_results.csv \\
      --output-dir analysis/plots/query_optimizer_hard_movies/runtime
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_optimizer_analysis import (  # noqa: E402
    RECALL_TARGET,
    plan_label,
    _present_algos,
)

# ANN indexes with a swept search hyperparameter (exclude brute-force).
DEFAULT_ALGOS = [
    "hnsw(faiss)",
    "hnsw(faiss)-post",
    "faiss-ivf",
    "faiss-ivf-post",
    "pgvector",
    "pgvector_ivf",
]

BF_ALGOS = ["faiss-flat", "pgvector_bf"]

# Combined 3-row figures: max ef_search, max n_probes, brute force.
MAX_VS_BF_GROUPS = {
    "faiss": [
        ("hnsw(faiss)", "ef_search"),
        ("faiss-ivf", "n_probes"),
        ("faiss-flat", "brute force"),
    ],
    "faiss_post": [
        ("hnsw(faiss)-post", "ef_search"),
        ("faiss-ivf-post", "n_probes"),
        ("faiss-flat", "brute force"),
    ],
    "pgvector": [
        ("pgvector", "ef_search"),
        ("pgvector_ivf", "n_probes"),
        ("pgvector_bf", "brute force"),
    ],
}

# Representative low / mid / high points on each starter grid.
DEFAULT_HNSW_HPS = [40, 100, 200]
DEFAULT_IVF_HPS = [20, 50, 100]

POINT_KEYS = ["query_id", "query_type", "filter_id", "algorithm", "hyperparam"]

# (stem, column, ylabel, log_scale, color, title_noun)
METRICS = (
    ("runtime", "runtime", "runtime (ms)", True, "#1f77b4", "runtime"),
    ("recall", "recall", "recall", False, "#d62728", "recall"),
    ("dist_comps", "dist_comps", "distance computations", True, "#2ca02c", "distance computations"),
)


def _algo_slug(algo: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", algo).strip("_")


def _hp_name_for_algo(algo: str, df: pd.DataFrame) -> str:
    sub = df.loc[df["algorithm"] == algo, "hyperparam_name"]
    if sub.empty:
        return "hyperparam"
    name = str(sub.iloc[0])
    return name if name and name != "none" else "hyperparam"


def _display_hp_name(name: str) -> str:
    return {"probes": "n_probes", "none": "brute force"}.get(name, name)


def _max_hp(df: pd.DataFrame, algo: str) -> int | None:
    hps = df.loc[df["algorithm"] == algo, "hyperparam"].dropna()
    if hps.empty:
        return None
    return int(hps.max())


def _algo_slice(df: pd.DataFrame, algo: str, hp: int | None = None) -> pd.DataFrame:
    sub = df[df["algorithm"] == algo]
    if hp is not None:
        sub = sub[sub["hyperparam"] == hp]
    return sub


def _row_stats(values: np.ndarray, as_ms: bool, count_metric: bool = False) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no data"
    med = float(np.median(finite))
    if as_ms:
        p95 = float(np.percentile(finite, 95))
        return f"med={med:.2f} ms, p95={p95:.2f} ms"
    if count_metric:
        p95 = float(np.percentile(finite, 95))
        return f"med={med:.0f}, p95={p95:.0f}"
    mean = float(np.mean(finite))
    frac = float(np.mean(finite >= RECALL_TARGET))
    return f"med={med:.3f}, mean={mean:.3f}, ≥{RECALL_TARGET:g}: {100 * frac:.1f}%"


def _row_title_prefix(algo: str, df: pd.DataFrame, hp: int | None, *, mark_max: bool = False) -> str:
    label = plan_label(algo)
    if algo in BF_ALGOS or hp is None:
        return f"brute force — {label}"
    hp_name = _display_hp_name(_hp_name_for_algo(algo, df))
    suffix = " (max)" if mark_max else ""
    return f"{hp_name}={hp}{suffix} — {label}"


def prepare_query_df(df: pd.DataFrame, k: int | None = 10) -> pd.DataFrame:
    """Optionally restrict to one top-k and collapse duplicate benchmark runs."""
    out = df.copy()
    if "split" in out.columns and (out["split"] == "test").any():
        out = out[out["split"] == "test"].copy()
    if k is not None and "k" in out.columns:
        out = out[out["k"] == k]
        if out.empty:
            return out

    group_keys = [c for c in POINT_KEYS if c in out.columns]
    if not out.duplicated(subset=group_keys).any():
        return out.reset_index(drop=True)
    agg: dict[str, str] = {
        "runtime": "mean",
        "recall": "mean",
        "dist_comps": "mean",
        "filter_selectivity": "first",
        "post_hardness": "first",
        "gls_correlation": "first",
        "hyperparam_name": "first",
    }
    present = {c: f for c, f in agg.items() if c in out.columns}
    return out.groupby(group_keys, as_index=False).agg(present)


def _metric_values(row: pd.DataFrame, column: str, as_ms: bool) -> np.ndarray:
    vals = row[column].to_numpy(dtype=float)
    if as_ms:
        vals = vals * 1000.0
    return vals


def _panel_hist(
    ax,
    values: np.ndarray,
    color: str,
    xlabel: str,
    log_scale: bool,
) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    if log_scale:
        lo = max(finite.min(), 1e-3)
        hi = max(finite.max(), lo * 1.01)
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)
        ax.set_xscale("log")
    else:
        lo = min(finite.min(), 0.0)
        hi = max(finite.max(), lo + 1e-6)
        # Keep recall histograms on a stable [0, 1]-ish domain.
        if xlabel == "recall":
            lo, hi = 0.0, 1.0
        bins = np.linspace(lo, hi, 41)
    ax.hist(finite, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(finite), color="black", lw=1.0, ls="--", label="median")
    if not log_scale and xlabel == "recall":
        ax.axvline(
            RECALL_TARGET,
            color="#555555",
            lw=1.0,
            ls=":",
            label=f"target {RECALL_TARGET:g}",
        )
        ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# queries")
    ax.legend(fontsize=7, loc="best", frameon=False)


def _panel_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    color: str,
    log_y: bool,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.scatter(
        x[mask],
        y[mask],
        s=10,
        c=color,
        alpha=0.35,
        edgecolors="none",
        rasterized=True,
    )
    if log_y:
        ax.set_yscale("log")
    elif ylabel == "recall":
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(RECALL_TARGET, color="#555555", lw=0.9, ls=":", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_metric_panel_grid(
    rows: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    stem: str,
    column: str,
    ylabel: str,
    log_scale: bool,
    color: str,
    title: str,
    share_metric_limits: bool = False,
) -> Path | None:
    """rows = [(row_title_prefix, dataframe)] with the metric already in ``column``."""
    use_rows = [(prefix, row) for prefix, row in rows if row is not None and not row.empty]
    if not use_rows:
        return None
    if column not in use_rows[0][1].columns:
        print(f"  skip {stem}: missing column {column}")
        return None
    if not any(np.isfinite(row[column].to_numpy(dtype=float)).any() for _, row in use_rows):
        print(f"  skip {stem}: {column} is empty")
        return None

    as_ms = column == "runtime"
    n_rows = len(use_rows)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(14, 3.1 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    plotted: list[np.ndarray] = []
    for r, (prefix, row) in enumerate(use_rows):
        values = _metric_values(row, column, as_ms=as_ms)
        plotted.append(values)
        sel = row["filter_selectivity"].to_numpy(dtype=float)
        post = row["post_hardness"].to_numpy(dtype=float)
        gls = row["gls_correlation"].to_numpy(dtype=float)
        stats = _row_stats(values, as_ms=as_ms, count_metric=(column == "dist_comps"))

        _panel_hist(axes[r, 0], values, color, ylabel, log_scale)
        _panel_scatter(axes[r, 1], sel, values, "filter selectivity σ", ylabel, color, log_scale)
        _panel_scatter(axes[r, 2], post, values, "Post_Hardness", ylabel, color, log_scale)
        _panel_scatter(axes[r, 3], gls, values, "GLS correlation ρ", ylabel, color, log_scale)
        axes[r, 0].set_title(f"{prefix}  (n={len(row)}, {stats})", fontsize=10, loc="left")

    if share_metric_limits:
        finite = np.concatenate([v[np.isfinite(v)] for v in plotted if v.size])
        if finite.size:
            if log_scale:
                lo = max(float(finite.min()), 1e-3)
                hi = max(float(finite.max()), lo * 1.01)
                pad = (hi / lo) ** 0.05
                lo, hi = lo / pad, hi * pad
                for r in range(n_rows):
                    axes[r, 0].set_xlim(lo, hi)
                    for c in (1, 2, 3):
                        axes[r, c].set_ylim(lo, hi)
            elif ylabel == "recall":
                for r in range(n_rows):
                    axes[r, 0].set_xlim(0.0, 1.0)
                    for c in (1, 2, 3):
                        axes[r, c].set_ylim(-0.02, 1.02)

    fig.suptitle(title, fontsize=13)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=150)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {png}")
    return png


def plot_overlay_cdf(
    series: list[tuple[str, np.ndarray]],
    output_dir: Path,
    stem: str,
    xlabel: str,
    log_scale: bool,
    title: str,
) -> Path | None:
    use = [(lab, v[np.isfinite(v)]) for lab, v in series]
    use = [(lab, np.sort(v)) for lab, v in use if v.size]
    if not use:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, (lab, values) in enumerate(use):
        y = np.arange(1, values.size + 1) / values.size
        ax.plot(
            values,
            y,
            color=cmap(i / max(len(use) - 1, 1)),
            lw=1.8,
            label=lab,
        )
    if log_scale:
        ax.set_xscale("log")
    elif xlabel == "recall":
        ax.set_xlim(0.0, 1.0)
        ax.axvline(RECALL_TARGET, color="#555555", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=150)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {png}")
    return png


def plot_algo_metric_panels(
    df: pd.DataFrame,
    algo: str,
    hps: list[int],
    output_dir: Path,
    metric_stem: str,
    column: str,
    ylabel: str,
    log_scale: bool,
    color: str,
    title_noun: str,
    table: str | None = None,
) -> Path | None:
    """One figure: rows = HPs, cols = dist / σ / Post_Hardness / ρ."""
    if column not in df.columns:
        print(f"  skip {metric_stem}/{algo}: missing column {column}")
        return None

    sub = df[df["algorithm"] == algo]
    if sub.empty:
        return None
    if not np.isfinite(sub[column].to_numpy(dtype=float)).any():
        print(f"  skip {metric_stem}/{algo}: {column} empty")
        return None

    hp_name = _hp_name_for_algo(algo, sub)
    available = sorted(int(h) for h in sub["hyperparam"].dropna().unique())
    use_hps = [h for h in hps if h in available]
    if not use_hps:
        print(f"  skip {algo}: none of {hps} present (have {available[:8]}...)")
        return None

    n_rows = len(use_hps)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(14, 3.1 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    label = plan_label(algo)
    table_bit = f", {table}" if table else ""
    as_ms = column == "runtime"

    for r, hp in enumerate(use_hps):
        row = sub[sub["hyperparam"] == hp]
        values = _metric_values(row, column, as_ms=as_ms)
        sel = row["filter_selectivity"].to_numpy(dtype=float)
        post = row["post_hardness"].to_numpy(dtype=float)
        gls = row["gls_correlation"].to_numpy(dtype=float)
        stats = _row_stats(values, as_ms=as_ms, count_metric=(column == "dist_comps"))

        _panel_hist(axes[r, 0], values, color, ylabel, log_scale)
        _panel_scatter(axes[r, 1], sel, values, "filter selectivity σ", ylabel, color, log_scale)
        _panel_scatter(axes[r, 2], post, values, "Post_Hardness", ylabel, color, log_scale)
        _panel_scatter(axes[r, 3], gls, values, "GLS correlation ρ", ylabel, color, log_scale)

        axes[r, 0].set_title(
            f"{hp_name}={hp}  (n={len(row)}, {stats})",
            fontsize=10,
            loc="left",
        )

    fig.suptitle(
        f"Per-query {title_noun} @ fixed search params — {label}{table_bit}",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metric_stem}_panels_{_algo_slug(algo)}"
    if table:
        stem = f"{stem}_{table}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")
    return png


def plot_hp_overlay_cdf(
    df: pd.DataFrame,
    algo: str,
    hps: list[int],
    output_dir: Path,
    metric_stem: str,
    column: str,
    xlabel: str,
    log_scale: bool,
    title_noun: str,
    table: str | None = None,
) -> Path | None:
    """Compact companion: overlaid CDFs for the selected HPs."""
    if column not in df.columns:
        return None
    sub = df[df["algorithm"] == algo]
    if sub.empty:
        return None
    if not np.isfinite(sub[column].to_numpy(dtype=float)).any():
        return None
    hp_name = _hp_name_for_algo(algo, sub)
    available = set(int(h) for h in sub["hyperparam"].dropna().unique())
    use_hps = [h for h in hps if h in available]
    if not use_hps:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    as_ms = column == "runtime"
    for i, hp in enumerate(use_hps):
        values = np.sort(
            _metric_values(sub.loc[sub["hyperparam"] == hp], column, as_ms=as_ms)
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        ax.plot(
            values,
            y,
            color=cmap(i / max(len(use_hps) - 1, 1)),
            lw=1.8,
            label=f"{hp_name}={hp}",
        )
    if log_scale:
        ax.set_xscale("log")
    elif xlabel == "recall":
        ax.set_xlim(0.0, 1.0)
        ax.axvline(RECALL_TARGET, color="#555555", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    label = plan_label(algo)
    table_bit = f", {table}" if table else ""
    ax.set_title(f"{title_noun.capitalize()} CDF @ fixed search params — {label}{table_bit}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metric_stem}_cdf_{_algo_slug(algo)}"
    if table:
        stem = f"{stem}_{table}"
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=150)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {png}")
    return png


def _stem(metric_stem: str, kind: str, table: str | None) -> str:
    stem = f"{metric_stem}_{kind}"
    if table:
        stem = f"{stem}_{table}"
    return stem


def plot_single_plan_panels(
    df: pd.DataFrame,
    algo: str,
    output_dir: Path,
    table: str | None,
    hp: int | None = None,
    stem_suffix: str = "",
    mark_max: bool = False,
) -> list[Path]:
    """One-row {recall|runtime}_panels for a single (algo, hp) — max HP or BF."""
    row = _algo_slice(df, algo, hp)
    if row.empty:
        return []
    prefix = _row_title_prefix(algo, df, hp, mark_max=mark_max)
    label = plan_label(algo)
    table_bit = f", {table}" if table else ""
    slug = _algo_slug(algo) + stem_suffix
    written: list[Path] = []
    for metric_stem, column, ylabel, log_scale, color, title_noun in METRICS:
        p = plot_metric_panel_grid(
            [(prefix, row)],
            output_dir,
            stem=_stem(metric_stem, f"panels_{slug}", table),
            column=column,
            ylabel=ylabel,
            log_scale=log_scale,
            color=color,
            title=f"Per-query {title_noun} @ fixed search params — {label}{table_bit}",
        )
        if p is not None:
            written.append(p)
    return written


def plot_max_vs_bf_panels(
    df: pd.DataFrame,
    output_dir: Path,
    table: str | None,
) -> list[Path]:
    """3-row comparison: max ef_search, max n_probes, brute force."""
    written: list[Path] = []
    table_bit = f", {table}" if table else ""
    present = set(df["algorithm"].unique())

    for group, specs in MAX_VS_BF_GROUPS.items():
        row_specs: list[tuple[str, pd.DataFrame, str]] = []
        for algo, kind in specs:
            if algo not in present:
                continue
            if kind == "brute force":
                hp = None
                row = _algo_slice(df, algo)
            else:
                hp = _max_hp(df, algo)
                if hp is None:
                    continue
                row = _algo_slice(df, algo, hp)
            if row.empty:
                continue
            prefix = _row_title_prefix(algo, df, hp, mark_max=(kind != "brute force"))
            row_specs.append((prefix, row, kind))
        if len(row_specs) < 2:
            print(f"  skip max-vs-bf/{group}: need ≥2 plans, have {len(row_specs)}")
            continue

        rows = [(p, r) for p, r, _ in row_specs]
        for metric_stem, column, ylabel, log_scale, color, title_noun in METRICS:
            p = plot_metric_panel_grid(
                rows,
                output_dir,
                stem=_stem(metric_stem, f"panels_max_vs_bf_{group}", table),
                column=column,
                ylabel=ylabel,
                log_scale=log_scale,
                color=color,
                title=(
                    f"Per-query {title_noun} @ max ef_search / max n_probes / "
                    f"brute force — {group.replace('_', ' ')}{table_bit}"
                ),
                share_metric_limits=True,
            )
            if p is not None:
                written.append(p)

            as_ms = column == "runtime"
            cdf_series = [
                (prefix.split(" — ")[0], _metric_values(row, column, as_ms=as_ms))
                for prefix, row, _ in row_specs
            ]
            p2 = plot_overlay_cdf(
                cdf_series,
                output_dir,
                stem=_stem(metric_stem, f"cdf_max_vs_bf_{group}", table),
                xlabel=ylabel,
                log_scale=log_scale,
                title=(
                    f"{title_noun.capitalize()} CDF @ max search params vs brute force"
                    f" — {group.replace('_', ' ')}{table_bit}"
                ),
            )
            if p2 is not None:
                written.append(p2)
    return written


def run_plots(
    df: pd.DataFrame,
    output_dir: Path,
    algos: list[str],
    hnsw_hps: list[int],
    ivf_hps: list[int],
    k: int | None = 10,
    include_fixed_hps: bool = True,
) -> list[Path]:
    prepared = prepare_query_df(df, k=k)
    if prepared.empty:
        print("No rows after prepare; nothing to plot.")
        return []

    table = None
    if "query_type" in prepared.columns:
        types = prepared["query_type"].dropna().unique()
        if len(types) == 1:
            table = str(types[0])

    present = _present_algos(prepared, algos)
    written: list[Path] = []
    if not include_fixed_hps:
        present_for_mid = []
    else:
        present_for_mid = present
    for algo in present_for_mid:
        hp_name = _hp_name_for_algo(algo, prepared)
        if hp_name == "probes" or "ivf" in algo:
            hps = ivf_hps
        else:
            hps = hnsw_hps
        print(f"Plotting {algo} at {hp_name} in {hps}")
        for metric_stem, column, ylabel, log_scale, color, title_noun in METRICS:
            p = plot_algo_metric_panels(
                prepared,
                algo,
                hps,
                output_dir,
                metric_stem=metric_stem,
                column=column,
                ylabel=ylabel,
                log_scale=log_scale,
                color=color,
                title_noun=title_noun,
                table=table,
            )
            if p is not None:
                written.append(p)
            p2 = plot_hp_overlay_cdf(
                prepared,
                algo,
                hps,
                output_dir,
                metric_stem=metric_stem,
                column=column,
                xlabel=ylabel,
                log_scale=log_scale,
                title_noun=title_noun,
                table=table,
            )
            if p2 is not None:
                written.append(p2)

    print("Plotting max-HP and brute-force panels")
    for algo in present:
        hp = _max_hp(prepared, algo)
        if hp is None:
            continue
        written.extend(
            plot_single_plan_panels(
                prepared, algo, output_dir, table, hp=hp, stem_suffix="_max", mark_max=True
            )
        )
    for algo in _present_algos(prepared, BF_ALGOS):
        written.extend(plot_single_plan_panels(prepared, algo, output_dir, table))

    print("Plotting max ef_search / max n_probes / brute-force comparison")
    written.extend(plot_max_vs_bf_panels(prepared, output_dir, table))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        required=True,
        help="Per-query results CSV (e.g. analysis/plots/query_optimizer_hard_movies/"
        "all_query_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <csv_dir>/runtime)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=DEFAULT_ALGOS,
        help="ANN algorithms to plot (default: FAISS/pgvector HNSW+IVF pre/post)",
    )
    parser.add_argument(
        "--hnsw-hps",
        nargs="*",
        type=int,
        default=DEFAULT_HNSW_HPS,
        help=f"ef_search values (default: {DEFAULT_HNSW_HPS})",
    )
    parser.add_argument(
        "--ivf-hps",
        nargs="*",
        type=int,
        default=DEFAULT_IVF_HPS,
        help=f"probes values (default: {DEFAULT_IVF_HPS})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k to keep (default 10; pass 0 to keep all k)",
    )
    parser.add_argument(
        "--max-vs-bf-only",
        action="store_true",
        help="Skip mid-range HP grids; write only max-HP, brute-force, and comparison panels",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_csv = Path(args.results_csv)
    if not results_csv.is_absolute():
        results_csv = root / results_csv
    if not results_csv.is_file():
        raise SystemExit(f"missing results CSV: {results_csv}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_csv.parent / "runtime"
    )
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    print(f"Loading {results_csv}")
    df = pd.read_csv(results_csv)
    needed = {"algorithm", "hyperparam", "runtime", "recall", "filter_selectivity"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")
    for col in ("post_hardness", "gls_correlation"):
        if col not in df.columns:
            print(f"warning: {col} absent; that scatter panel will be empty")
            df[col] = np.nan

    k = None if args.k == 0 else args.k
    written = run_plots(
        df,
        output_dir,
        algos=list(args.algorithms),
        hnsw_hps=list(args.hnsw_hps),
        ivf_hps=list(args.ivf_hps),
        k=k,
        include_fixed_hps=not args.max_vs_bf_only,
    )
    if not written:
        raise SystemExit("no runtime/recall plots written")
    print(f"Done: {len(written)} files under {output_dir}")


if __name__ == "__main__":
    main()
