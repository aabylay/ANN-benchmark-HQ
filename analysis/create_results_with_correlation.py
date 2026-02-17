#!/usr/bin/env python3
"""
Script to create a CSV file with ANN benchmark results and GLS correlation values,
and optionally generate correlation analysis plots.

This script:
1. Reads existing results from all_results.csv
2. Matches queries with filter statistics to get correlation values
3. Outputs a new CSV with correlation columns added
4. Optionally generates plots for correlation analysis
"""

import os
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional

# Plotting imports (optional)
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    # Set matplotlib to use DejaVu Serif (LaTeX-like font) and configure mathtext
    mpl.rcParams['lines.linewidth'] = 2
    plt.rc('font', family='serif', serif='DejaVu Serif', size=24)
    plt.rc('mathtext', default='regular')
except ImportError:
    PLOTTING_AVAILABLE = False


# Plotting configuration

# Correlation level colors
COLORS = {
    'high': '#2ecc71',      # Green for positive correlation
    'average': '#3498db',   # Blue for neutral
    'low': '#e74c3c',       # Red for negative correlation
}

# System-based colors: FAISS=green, Milvus=red, pgvector=blue
# IVF variants are a few tones darker
SYSTEM_COLORS = {
    'FAISS': {'HNSW': '#2ecc71', 'IVF': '#1a7a42'},
    'Milvus': {'HNSW': '#FF4444', 'IVF': '#AA0000'},
    'pgvector': {'HNSW': '#4488FF', 'IVF': '#0000AA'},
}

# Algorithm to system mapping
ALGO_TO_SYSTEM = {
    'hnsw(faiss)': 'FAISS',
    'faiss-ivf': 'FAISS',
    'milvus-hnsw': 'Milvus',
    'milvus-ivfflat': 'Milvus',
    'pgvector': 'pgvector',
    'pgvector_ivf': 'pgvector',
}

# System algorithm lists
SYSTEM_ALGOS = {
    'FAISS': {'HNSW': 'hnsw(faiss)', 'IVF': 'faiss-ivf'},
    'Milvus': {'HNSW': 'milvus-hnsw', 'IVF': 'milvus-ivfflat'},
    'pgvector': {'HNSW': 'pgvector', 'IVF': 'pgvector_ivf'},
}


def load_filter_stats(filter_stats_path: str) -> pd.DataFrame:
    """Load and preprocess filter statistics CSV."""
    df = pd.read_csv(filter_stats_path)
    return df


def find_closest_filter(
    query_type: str, 
    target_selectivity: float, 
    filter_stats: pd.DataFrame,
    tolerance: float = 0.05,
    preferred_attribute: Optional[str] = None
) -> Optional[str]:
    """
    Find the filter that most closely matches the target selectivity for a given query type.
    """
    qt_df = filter_stats[filter_stats['query_type'] == query_type]
    filter_sels = qt_df.groupby('filter')['selectivity'].first()
    
    # Filter by preferred attribute if specified
    if preferred_attribute:
        filter_sels = filter_sels[filter_sels.index.str.contains(preferred_attribute)]
        if len(filter_sels) == 0:
            return None
    
    # Find closest match
    diffs = np.abs(filter_sels - target_selectivity)
    min_diff_idx = diffs.idxmin()
    min_diff = diffs[min_diff_idx]
    
    if min_diff <= tolerance:
        return min_diff_idx
    return None


def get_query_type_from_filename(filename: str) -> Optional[str]:
    """Determine query type from HDF5 filename."""
    if 'movies' in filename.lower():
        return 'flex_movies_sim'
    elif 'reviews' in filename.lower():
        return 'flex_reviews_sim'
    return None


def parse_hdf5_config(filename: str, algorithm: str) -> Dict:
    """
    Parse config parameters from HDF5 filename.
    
    HNSW: movies_angular_M_{m}_efConstruction_{ef_construct}_{ef_search}.hdf5
    IVF: movies_angular_clusters_{clusters}_{probes}.hdf5
    """
    import re
    
    config = {}
    algo_lower = algorithm.lower()
    
    if 'hnsw' in algo_lower or algorithm == 'pgvector':
        # HNSW pattern: M_{m}_efConstruction_{ef_construct}_{ef_search}
        match = re.search(r'M_(\d+)_efConstruction_(\d+)_(\d+)', filename)
        if match:
            config['m'] = int(match.group(1))
            config['ef_search'] = int(match.group(3))
    elif 'ivf' in algo_lower:
        # IVF pattern: clusters_{clusters}_{probes} (faiss/pgvector)
        match = re.search(r'clusters_(\d+)_(\d+)', filename)
        if match:
            config['clusters'] = int(match.group(1))
            config['probes'] = int(match.group(2))
        else:
            # Milvus IVF pattern: {dim}_nlist_{nlist}_{nprobe}
            match = re.search(r'_nlist_(\d+)_(\d+)', filename)
            if match:
                config['clusters'] = int(match.group(1))
                config['probes'] = int(match.group(2))
    
    return config


def create_results_with_correlation(
    results_dir: str,
    filter_stats: pd.DataFrame,
    fid: int,
    k: int = 10,
    movies_filter_attr: Optional[str] = None,
    reviews_filter_attr: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a DataFrame with results and correlation values.
    
    Returns:
        DataFrame with columns: query_id, algorithm, config, recall, runtime, 
                               correlation_acorn, correlation_gls, filter_name
    """
    # Get selectivity for this filter ID
    all_results_path = os.path.join(results_dir, 'all_results.csv')
    all_results = pd.read_csv(all_results_path)
    fid_results = all_results[all_results['filter_id'] == fid]
    
    if len(fid_results) == 0:
        print(f"No results found for fid{fid}")
        return pd.DataFrame()
    
    target_selectivity = fid_results['filter_selectivity'].iloc[0]
    print(f"Filter ID {fid} has selectivity {target_selectivity:.6f}")
    
    # Find path to fid directory
    fid_dir = os.path.join(results_dir, f'fid{fid}', str(k))
    if not os.path.exists(fid_dir):
        print(f"Directory not found: {fid_dir}")
        return pd.DataFrame()
    
    # Map query type to preferred attribute
    preferred_attrs = {
        'flex_movies_sim': movies_filter_attr,
        'flex_reviews_sim': reviews_filter_attr
    }
    
    # Build correlation lookup for each query type
    correlation_lookup = {}
    for query_type in ['flex_movies_sim', 'flex_reviews_sim']:
        filter_name = find_closest_filter(
            query_type, target_selectivity, filter_stats,
            preferred_attribute=preferred_attrs.get(query_type)
        )
        if filter_name:
            mask = (filter_stats['query_type'] == query_type) & (filter_stats['filter'] == filter_name)
            subset = filter_stats[mask].set_index('q_id')
            correlation_lookup[query_type] = {
                'filter_name': filter_name,
                'acorn': subset['correlation_ACORN'].to_dict(),
                'gls': subset['correlation_NEW'].to_dict()
            }
            print(f"  {query_type} -> {filter_name}")
    
    # Determine which query IDs have correlation data
    available_query_ids = set()
    for query_type, lookup in correlation_lookup.items():
        available_query_ids.update(lookup['acorn'].keys())
    
    if not available_query_ids:
        print("Warning: No correlation data available!")
        return pd.DataFrame()
    
    max_query_id = max(available_query_ids)
    print(f"  Correlation data available for query_ids 0-{max_query_id} ({len(available_query_ids)} queries)")
    
    # Collect results from all HDF5 files (only for queries with correlation data)
    results_list = []
    
    for algo_name in os.listdir(fid_dir):
        algo_dir = os.path.join(fid_dir, algo_name)
        if not os.path.isdir(algo_dir):
            continue
        
        for hdf5_file in os.listdir(algo_dir):
            if not hdf5_file.endswith('.hdf5'):
                continue
            
            hdf5_path = os.path.join(algo_dir, hdf5_file)
            query_type = get_query_type_from_filename(hdf5_file)
            
            if query_type is None or query_type not in correlation_lookup:
                continue
            
            lookup = correlation_lookup[query_type]
            filter_name = lookup['filter_name']
            
            # Extract config from filename (remove prefix and .hdf5)
            config = hdf5_file.replace('.hdf5', '')
            
            # Read HDF5 file
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    times = f['times'][:]
                    num_queries = len(times)
                    
                    # Get distances if available (for computing recall if needed)
                    # For now, we'll need to get recall from all_results.csv
            except Exception as e:
                print(f"  Error reading {hdf5_path}: {e}")
                continue
            
            # Parse config parameters from filename
            config_params = parse_hdf5_config(hdf5_file, algo_name)
            
            # Create per-query results (only for queries with correlation data)
            for q_id in range(min(num_queries, max_query_id + 1)):
                if q_id not in lookup['acorn']:
                    continue
                    
                acorn_corr = lookup['acorn'].get(q_id, np.nan)
                gls_corr = lookup['gls'].get(q_id, np.nan)
                
                row = {
                    'query_id': q_id,
                    'query_type': query_type,
                    'filter_id': fid,
                    'filter_name': filter_name,
                    'filter_selectivity': target_selectivity,
                    'algorithm': algo_name,
                    'config': config,
                    'runtime': times[q_id],
                    'correlation_acorn': acorn_corr,
                    'correlation_gls': gls_corr
                }
                # Add config params for recall matching
                row.update(config_params)
                results_list.append(row)
    
    df = pd.DataFrame(results_list)
    return df


def add_recall_from_all_results(
    results_df: pd.DataFrame,
    results_dir: str,
    fid: int
) -> pd.DataFrame:
    """
    Add recall values from all_results_hnsw.csv and all_results_ivf.csv to the results DataFrame.
    Matches recall by config parameters (m, ef_search for HNSW; clusters, probes for IVF).
    """
    hnsw_path = os.path.join(results_dir, 'all_results_hnsw.csv')
    ivf_path = os.path.join(results_dir, 'all_results_ivf.csv')
    
    # Initialize recall column
    results_df = results_df.copy()
    results_df['recall'] = np.nan
    
    # Helper: map query_type from results_df to prefix in all_results CSVs
    def _query_type_to_prefix(qt: str) -> str:
        """Map 'flex_movies_sim' -> 'qm', 'flex_reviews_sim' -> 'qr'."""
        if 'movies' in qt:
            return 'qm'
        elif 'reviews' in qt:
            return 'qr'
        return 'q'
    
    # Process HNSW results
    if os.path.exists(hnsw_path):
        hnsw_results = pd.read_csv(hnsw_path)
        hnsw_results = hnsw_results[hnsw_results['filter_id'] == fid].copy()
        
        # Extract query type prefix (qm/qr) and query_id number
        extracted = hnsw_results['query_id'].str.extract(r'(q[mr])(\d+)')
        hnsw_results['query_prefix'] = extracted[0]
        hnsw_results['query_id_num'] = pd.to_numeric(extracted[1], errors='coerce').fillna(-1).astype(int)
        
        # Create lookup key: prefix + query_id + algorithm + m + ef_search
        hnsw_results['lookup_key'] = (
            hnsw_results['query_prefix'] + '_' +
            hnsw_results['query_id_num'].astype(str) + '_' + 
            hnsw_results['algorithm'] + '_' +
            hnsw_results['m'].astype(str) + '_' +
            hnsw_results['ef_search'].astype(str)
        )
        # Average recall across duplicates (different efConstruction values share the same key)
        hnsw_recall_lookup = hnsw_results.groupby('lookup_key')['recall'].mean().to_dict()
        
        # Match HNSW algorithms in results_df
        hnsw_mask = results_df['algorithm'].apply(lambda x: 'hnsw' in x.lower() or x == 'pgvector')
        if hnsw_mask.any():
            hnsw_df = results_df[hnsw_mask].copy()
            hnsw_df['query_prefix'] = hnsw_df['query_type'].apply(_query_type_to_prefix)
            hnsw_df['lookup_key'] = (
                hnsw_df['query_prefix'] + '_' +
                hnsw_df['query_id'].astype(str) + '_' + 
                hnsw_df['algorithm'] + '_' +
                hnsw_df['m'].fillna(0).astype(int).astype(str) + '_' +
                hnsw_df['ef_search'].fillna(0).astype(int).astype(str)
            )
            results_df.loc[hnsw_mask, 'recall'] = hnsw_df['lookup_key'].map(hnsw_recall_lookup)
        
        print(f"  HNSW recall lookup: {len(hnsw_recall_lookup)} entries")
    
    # Process IVF results
    if os.path.exists(ivf_path):
        ivf_results = pd.read_csv(ivf_path)
        ivf_results = ivf_results[ivf_results['filter_id'] == fid].copy()
        
        # Extract query type prefix (qm/qr) and query_id number
        extracted = ivf_results['query_id'].str.extract(r'(q[mr])(\d+)')
        ivf_results['query_prefix'] = extracted[0]
        ivf_results['query_id_num'] = pd.to_numeric(extracted[1], errors='coerce').fillna(-1).astype(int)
        
        # Create lookup key: prefix + query_id + algorithm + clusters + probes
        ivf_results['lookup_key'] = (
            ivf_results['query_prefix'] + '_' +
            ivf_results['query_id_num'].astype(str) + '_' + 
            ivf_results['algorithm'] + '_' +
            ivf_results['clusters'].astype(str) + '_' +
            ivf_results['probes'].astype(str)
        )
        # Average recall across duplicates (different cluster counts may share the same key)
        ivf_recall_lookup = ivf_results.groupby('lookup_key')['recall'].mean().to_dict()
        
        # Match IVF algorithms in results_df
        ivf_mask = results_df['algorithm'].apply(lambda x: 'ivf' in x.lower())
        if ivf_mask.any():
            ivf_df = results_df[ivf_mask].copy()
            ivf_df['query_prefix'] = ivf_df['query_type'].apply(_query_type_to_prefix)
            ivf_df['lookup_key'] = (
                ivf_df['query_prefix'] + '_' +
                ivf_df['query_id'].astype(str) + '_' + 
                ivf_df['algorithm'] + '_' +
                ivf_df['clusters'].fillna(0).astype(int).astype(str) + '_' +
                ivf_df['probes'].fillna(0).astype(int).astype(str)
            )
            results_df.loc[ivf_mask, 'recall'] = ivf_df['lookup_key'].map(ivf_recall_lookup)
        
        print(f"  IVF recall lookup: {len(ivf_recall_lookup)} entries")
    
    # Report matching stats
    matched = results_df['recall'].notna().sum()
    total = len(results_df)
    print(f"  Recall matched: {matched}/{total} ({100*matched/total:.1f}%)")
    
    return results_df


def get_index_type(algorithm: str) -> str:
    """Classify algorithm into HNSW or IVF index type."""
    algo_lower = algorithm.lower()
    if 'hnsw' in algo_lower or algorithm == 'pgvector':
        return 'HNSW'
    elif 'ivf' in algo_lower:
        return 'IVF'
    else:
        return 'Other'


def categorize_correlation(corr: float, high_thresh: float = 0.3, low_thresh: float = -0.3) -> str:
    """Categorize correlation into high/average/low."""
    if pd.isna(corr):
        return 'average'
    if corr > high_thresh:
        return 'high'
    elif corr < low_thresh:
        return 'low'
    else:
        return 'average'


def darken_color(hex_color: str, factor: float = 0.5) -> str:
    """Darken a hex color by a factor (0=black, 1=unchanged)."""
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(hex_color)
    dark_rgb = tuple(c * factor for c in rgb)
    return mcolors.to_hex(dark_rgb)


def plot_scatter_correlation_vs_metric(
    df: pd.DataFrame,
    correlation_col: str,
    metric_col: str,
    ax: plt.Axes,
    title: str,
    system: str = None,
    by_index_type: bool = False,
    alpha: float = 0.3
):
    """Scatter plot of correlation vs a metric (recall or latency)."""
    df = df.copy()
    df = df.dropna(subset=[correlation_col, metric_col])
    
    if by_index_type:
        df['index_type'] = df['algorithm'].apply(get_index_type)
        
        for idx_type in ['HNSW', 'IVF']:
            type_data = df[df['index_type'] == idx_type]
            if len(type_data) == 0:
                continue
            
            # Use system-based colors if system is specified
            if system and system in SYSTEM_COLORS:
                color = SYSTEM_COLORS[system][idx_type]
            else:
                color = SYSTEM_COLORS.get('FAISS', {}).get(idx_type, 'steelblue')
            
            ax.scatter(type_data[correlation_col], type_data[metric_col], 
                      alpha=alpha, s=10, c=color, label=idx_type)
            
            # Compute trend line - thicker and darker for visibility
            if len(type_data) > 10:
                z = np.polyfit(type_data[correlation_col], type_data[metric_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(type_data[correlation_col].min(), type_data[correlation_col].max(), 100)
                trend_color = darken_color(color, 0.4)
                ax.plot(x_line, p(x_line), '-', color=trend_color, alpha=1.0, linewidth=4)
    else:
        ax.scatter(df[correlation_col], df[metric_col], alpha=alpha, s=10, c='steelblue')
        z = np.polyfit(df[correlation_col], df[metric_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[correlation_col].min(), df[correlation_col].max(), 100)
        ax.plot(x_line, p(x_line), '-', color='black', alpha=1.0, linewidth=4, label=f'Trend (slope={z[0]:.3f})')
    
    ax.set_xlabel(correlation_col.replace('_', ' ').title())
    ylabel = 'Latency (s)' if metric_col == 'latency' else metric_col.title()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=18)
    ax.grid(True, which="both", ls="--", zorder=0)


def plot_qps_recall_by_correlation(
    df: pd.DataFrame,
    correlation_col: str,
    ax: plt.Axes,
    title: str,
    system: str,
    query_type: str,
    high_thresh: float = 0.3,
    low_thresh: float = -0.3
):
    """
    Plot QPS vs Recall curves grouped by correlation level for a specific system.
    Both HNSW and IVF are overlaid: HNSW in system color, IVF in darker tone.
    High correlation lines are thicker, low correlation thinner.
    """
    # System-based colors (matching make_plots_hnsw_vs_ivf_comparison)
    system_colors_hnsw = {
        'FAISS': '#00FF00',     # Green
        'Milvus': '#FF0000',    # Red
        'pgvector': '#0000FF',  # Blue
    }
    system_colors_ivf = {
        'FAISS': '#66AA66',     # Darker green-grey
        'Milvus': '#AA6666',    # Darker red-grey
        'pgvector': '#6666AA',  # Darker blue-grey
    }
    
    # Linewidth by correlation level: high=thick, average=medium, low=thin
    corr_linewidths = {
        'high': 3.5,
        'average': 2.0,
        'low': 1.0,
    }
    corr_markers = {
        'high': 'o',
        'average': 's',
        'low': '+',
    }
    corr_markersizes = {
        'high': 6,
        'average': 6,
        'low': 4,
    }
    
    df = df.copy()
    # Filter to query type
    qt_map = {'movies': 'flex_movies_sim', 'reviews': 'flex_reviews_sim'}
    qt_full = qt_map.get(query_type, query_type)
    df = df[df['query_type'] == qt_full]
    
    # Drop rows with NaN recall or runtime
    df = df.dropna(subset=['recall', 'runtime'])
    if len(df) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=18)
        return
    
    df['corr_level'] = df[correlation_col].apply(
        lambda x: categorize_correlation(x, high_thresh, low_thresh)
    )
    df['qps'] = 1.0 / df['runtime']
    
    has_data = False
    
    for idx_type, linestyle in [('HNSW', '-'), ('IVF', '--')]:
        if system not in SYSTEM_ALGOS or idx_type not in SYSTEM_ALGOS[system]:
            continue
        
        algo_name = SYSTEM_ALGOS[system][idx_type]
        idx_df = df[df['algorithm'] == algo_name]
        if len(idx_df) == 0:
            continue
        
        color = system_colors_hnsw[system] if idx_type == 'HNSW' else system_colors_ivf[system]
        
        # Group by search-time parameter
        if idx_type == 'HNSW':
            group_cols = ['ef_search']
        else:
            group_cols = ['probes']
        group_cols = [c for c in group_cols if c in idx_df.columns]
        if not group_cols:
            group_cols = ['config']
        
        for level in ['high', 'average', 'low']:
            level_df = idx_df[idx_df['corr_level'] == level]
            if len(level_df) == 0:
                continue
            
            agg_df = level_df.groupby(group_cols).agg({
                'recall': 'mean',
                'qps': 'mean'
            }).reset_index().dropna().sort_values('recall')
            
            if len(agg_df) == 0:
                continue
            
            has_data = True
            count = len(level_df['query_id'].unique())
            lw = corr_linewidths[level]
            marker = corr_markers[level]
            label = f'{idx_type} {level.title()} ({count}q)'
            ax.plot(agg_df['recall'], agg_df['qps'], 
                    marker=marker, markersize=corr_markersizes[level], color=color,
                    linestyle=linestyle, linewidth=lw,
                    label=label, alpha=0.85)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    if has_data:
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=18)
    ax.grid(True, which="both", ls="--", zorder=0)


def plot_correlation_distribution(df: pd.DataFrame, ax: plt.Axes):
    """Plot histogram of ACORN and GLS correlations."""
    unique_corrs = df.groupby('query_id').first()[['correlation_acorn', 'correlation_gls']]
    ax.hist(unique_corrs['correlation_acorn'].dropna(), bins=30, alpha=0.6, label='Distance', color='orange')
    ax.hist(unique_corrs['correlation_gls'].dropna(), bins=30, alpha=0.6, label='GLS', color='purple')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Correlations')
    ax.legend(fontsize=22)
    ax.grid(True, which="both", ls="--", zorder=0)


def plot_algorithm_correlation_impact(
    df: pd.DataFrame,
    correlation_col: str,
    ax: plt.Axes,
    title: str,
    high_thresh: float = 0.3,
    low_thresh: float = -0.3
):
    """Bar plot showing recall for each algorithm at different correlation levels."""
    df = df.copy()
    df['corr_level'] = df[correlation_col].apply(
        lambda x: categorize_correlation(x, high_thresh, low_thresh)
    )
    
    results = []
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        results.append({
            'algorithm': algo,
            'high': algo_df[algo_df['corr_level'] == 'high']['recall'].mean(),
            'average': algo_df[algo_df['corr_level'] == 'average']['recall'].mean(),
            'low': algo_df[algo_df['corr_level'] == 'low']['recall'].mean(),
        })
    
    results_df = pd.DataFrame(results).sort_values('high', ascending=False, na_position='last')
    x = np.arange(len(results_df))
    width = 0.25
    
    ax.bar(x - width, results_df['high'], width, label='High Corr', color=COLORS['high'], alpha=0.8)
    ax.bar(x, results_df['average'], width, label='Avg Corr', color=COLORS['average'], alpha=0.8)
    ax.bar(x + width, results_df['low'], width, label='Low Corr', color=COLORS['low'], alpha=0.8)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean Recall')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['algorithm'], rotation=45, ha='right')
    ax.legend(fontsize=18)
    ax.grid(True, which="both", ls="--", zorder=0)


def generate_plots(df: pd.DataFrame, output_dir: str, high_thresh: float = 0.3, low_thresh: float = -0.3, 
                   algorithms: list = None):
    """Generate all correlation analysis plots."""
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib/seaborn not available. Skipping plots.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = df.copy()
    
    # Filter by algorithms if specified
    if algorithms:
        df = df[df['algorithm'].isin(algorithms)]
        print(f"  Filtered to algorithms: {algorithms}")
        print(f"  Rows after filter: {len(df)}")
    
    df['latency'] = df['runtime']
    
    # Get unique filter IDs - these should be sorted by selectivity (3=~1%, 5=~5%, 6=~10%)
    fids = sorted(df['filter_id'].unique())
    fids_to_plot = fids
    
    # Get selectivity labels
    sel_labels = {}
    for fid in fids_to_plot:
        fid_data = df[df['filter_id'] == fid]
        if len(fid_data) > 0:
            sel = fid_data['filter_selectivity'].iloc[0]
            sel_labels[fid] = f"~{sel:.0%}" if sel >= 0.01 else f"~{sel:.1%}"
    
    # Scatter plots by system: FAISS, Milvus, pgvector
    systems_scatter = [
        ('FAISS', ['hnsw(faiss)', 'faiss-ivf']),
        ('Milvus', ['milvus-hnsw', 'milvus-ivfflat']),
        ('pgvector', ['pgvector', 'pgvector_ivf']),
    ]
    
    for system_name, system_algos in systems_scatter:
        system_df = df[df['algorithm'].isin(system_algos)]
        
        if len(system_df) == 0:
            continue
        
        n_fids = len(fids_to_plot)
        fig, axes = plt.subplots(n_fids, 2, figsize=(16, 6 * n_fids))
        if n_fids == 1:
            axes = axes.reshape(1, -1)
        
        for i, fid in enumerate(fids_to_plot):
            fid_df = system_df[system_df['filter_id'] == fid]
            sel_label = sel_labels.get(fid, '')
            
            plot_scatter_correlation_vs_metric(fid_df, 'correlation_gls', 'recall', axes[i, 0], 
                f'{system_name}: GLS vs Recall ($\\sigma_g${sel_label})',
                system=system_name, by_index_type=True)
            plot_scatter_correlation_vs_metric(fid_df, 'correlation_gls', 'latency', axes[i, 1], 
                f'{system_name}: GLS vs Latency ($\\sigma_g${sel_label})',
                system=system_name, by_index_type=True)
        
        fig.tight_layout()
        fname = f'scatter_{system_name.lower()}_by_selectivity.png'
        fig.savefig(output_path / fname, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname}")
    
    # QPS-Recall curves by correlation level: 2x3 grid
    # Rows: Movies, Reviews; Columns: FAISS, Milvus, pgvector
    # HNSW and IVF overlaid in each subplot (IVF = darker tones, dashed)
    systems = ['FAISS', 'Milvus', 'pgvector']
    query_types = ['movies', 'reviews']
    
    fig3, axes3 = plt.subplots(len(query_types), len(systems), figsize=(24, 12))
    
    for row, qt in enumerate(query_types):
        for col, system_name in enumerate(systems):
            plot_qps_recall_by_correlation(
                df, 'correlation_gls', axes3[row, col],
                f'{system_name}, {qt.capitalize()}',
                system=system_name, query_type=qt,
                high_thresh=high_thresh, low_thresh=low_thresh
            )
    
    # Create single shared legend with grouped handles (3 system lines per entry)
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    
    legend_handles = []
    legend_labels = []
    
    _sys_hnsw = {'FAISS': '#00FF00', 'Milvus': '#FF0000', 'pgvector': '#0000FF'}
    _sys_ivf = {'FAISS': '#66AA66', 'Milvus': '#AA6666', 'pgvector': '#6666AA'}
    _lw = {'high': 3.5, 'average': 2.0, 'low': 1.0}
    _mk = {'high': 'o', 'average': 's', 'low': '+'}
    _ms = {'high': 6, 'average': 6, 'low': 4}
    
    for idx_type, ls in [('HNSW', '-'), ('IVF', '--')]:
        colors = _sys_hnsw if idx_type == 'HNSW' else _sys_ivf
        for level in ['high', 'average', 'low']:
            handles = tuple(
                Line2D([0], [0], color=colors[s], linestyle=ls,
                       linewidth=_lw[level], marker=_mk[level],
                       markersize=_ms[level])
                for s in systems
            )
            legend_handles.append(handles)
            if level == 'high': legend_label = f'{idx_type}, High GLS Correlation: $\\rho_q > 0.3$'
            elif level == 'average': legend_label = f'{idx_type}, Average GLS Correlation: $\\rho_q \\in [-0.3, 0.3]$'
            elif level == 'low': legend_label = f'{idx_type}, Low GLS Correlation: $\\rho_q < -0.3$'
            legend_labels.append(legend_label)
    
    fig3.legend(legend_handles, legend_labels,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                loc='lower center', ncol=2, fontsize=24,
                handlelength=5, bbox_to_anchor=(0.5, -0.1))
    
    fig3.tight_layout(rect=[0, 0.06, 1, 1])
    fig3.savefig(output_path / 'qps_recall_by_correlation.png', dpi=150, bbox_inches='tight')
    fig3.savefig(output_path / 'qps_recall_by_correlation.pdf', dpi=150, bbox_inches='tight')
    print(f"  Saved: qps_recall_by_correlation.png")
    
    # Correlation distribution
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    plot_correlation_distribution(df, ax4)
    fig4.tight_layout()
    fig4.savefig(output_path / 'correlation_distribution.png', dpi=150, bbox_inches='tight')
    fig4.savefig(output_path / 'correlation_distribution.pdf', dpi=150, bbox_inches='tight')
    print(f"  Saved: correlation_distribution.png")
    
    # Algorithm impact
    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 7))
    plot_algorithm_correlation_impact(df, 'correlation_gls', ax5, 
        'Recall by GLS Correlation Level', high_thresh, low_thresh)
    fig5.tight_layout()
    fig5.savefig(output_path / 'algorithm_correlation_impact.png', dpi=150, bbox_inches='tight')
    fig5.savefig(output_path / 'algorithm_correlation_impact.pdf', dpi=150, bbox_inches='tight')
    print(f"  Saved: algorithm_correlation_impact.png")
    
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='Create CSV with ANN benchmark results and GLS correlation values'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/MoRe_UPD_large_attidx_0',
        help='Results directory path'
    )
    parser.add_argument(
        '--filter-stats',
        type=str,
        default='data/datasets/MoRe_large/queries/filter_stats_0_k2048.csv',
        help='Filter statistics CSV path'
    )
    parser.add_argument(
        '--fid',
        type=int,
        nargs='+',
        default=[3, 5, 6],
        help='Filter ID(s) to process (default: 3 5 6 for ~1%%, ~5%%, ~10%% selectivity). Can specify multiple: --fid 3 5 6'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='k value for results (default: 10)'
    )
    parser.add_argument(
        '--movies-filter',
        type=str,
        default=None,
        help='Preferred filter attribute for movies (e.g., avg_rating, num_votes)'
    )
    parser.add_argument(
        '--reviews-filter',
        type=str,
        default=None,
        help='Preferred filter attribute for reviews (e.g., total_votes)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: results_with_correlation_fid{fid}.csv)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate correlation analysis plots'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: analysis/plots/correlation_UPD_fid{fid})'
    )
    parser.add_argument(
        '--high-thresh',
        type=float,
        default=0.3,
        help='Threshold for high correlation (default: 0.3)'
    )
    parser.add_argument(
        '--low-thresh',
        type=float,
        default=-0.3,
        help='Threshold for low correlation (default: -0.3)'
    )
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        default=None,
        help='Filter to specific algorithms (e.g., --algorithms "hnsw(faiss)" "faiss-ivf")'
    )
    
    args = parser.parse_args()
    
    fid_str = '_'.join(map(str, args.fid)) if len(args.fid) > 1 else str(args.fid[0])
    
    if args.output is None:
        args.output = os.path.join(args.results_dir, f'results_with_correlation_fid{fid_str}.csv')
    
    if args.plot_dir is None:
        args.plot_dir = f'analysis/plots/correlation_fid{fid_str}'
    
    print("=" * 60)
    print("Creating Results CSV with GLS Correlation")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Filter stats: {args.filter_stats}")
    print(f"Filter IDs: {args.fid}")
    print(f"k: {args.k}")
    print(f"Movies filter preference: {args.movies_filter or '(auto)'}")
    print(f"Reviews filter preference: {args.reviews_filter or '(auto)'}")
    print(f"Output: {args.output}")
    print()
    
    # Load filter stats
    filter_stats = load_filter_stats(args.filter_stats)
    print(f"Loaded {len(filter_stats)} filter-query combinations")
    
    # Process each fid and combine results
    all_results = []
    for fid in args.fid:
        print(f"\nProcessing fid{fid}...")
        fid_results = create_results_with_correlation(
            args.results_dir,
            filter_stats,
            fid,
            args.k,
            movies_filter_attr=args.movies_filter,
            reviews_filter_attr=args.reviews_filter
        )
        
        if len(fid_results) > 0:
            # Add recall values
            fid_results = add_recall_from_all_results(fid_results, args.results_dir, fid)
            all_results.append(fid_results)
    
    if not all_results:
        print("No results generated!")
        return
    
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\nCombined results from {len(args.fid)} filter IDs")
    
    # Reorder columns
    cols = ['query_id', 'query_type', 'filter_id', 'filter_name', 'filter_selectivity',
            'algorithm', 'config', 'm', 'ef_search', 'clusters', 'probes',
            'recall', 'runtime', 'correlation_acorn', 'correlation_gls']
    results_df = results_df[[c for c in cols if c in results_df.columns]]
    
    # Save to CSV
    results_df.to_csv(args.output, index=False)
    
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"  Total rows: {len(results_df)}")
    print(f"  Unique queries: {results_df['query_id'].nunique()}")
    print(f"  Unique algorithms: {results_df['algorithm'].nunique()}")
    print(f"  Unique configs: {results_df['config'].nunique()}")
    print(f"  Output saved to: {args.output}")
    
    # Show sample
    print(f"\nSample rows:")
    print(results_df.head(10).to_string())
    
    # Generate plots if requested
    if args.plot:
        print(f"\n{'=' * 60}")
        print("Generating plots...")
        print("=" * 60)
        generate_plots(results_df, args.plot_dir, args.high_thresh, args.low_thresh, args.algorithms)
        print(f"\nPlots saved to: {args.plot_dir}")


if __name__ == '__main__':
    main()
