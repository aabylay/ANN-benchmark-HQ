import pandas as pd
import numpy as np
import os

def compute_averages(df):
    """Compute averages grouped by relevant columns"""
    df['query_type'] = df['query_id'].apply(lambda x: 'movies' if x.startswith('qm') else 'reviews')
    averages = df.groupby(['query_type', 'filter_id', 'filter_selectivity', 'k', 'ef_search', 'algorithm', 'm'])[['recall', 'runtime']].mean().reset_index()
    averages['throughput'] = 1 / averages['runtime']
    return averages

def analyze_recall_comparison(dataset_size='small', algorithm='pgvector', query_type='movies'):
    """
    Analyze recall and QPS comparison between att_idx=0 and att_idx=1
    
    Returns a dataframe with columns:
    - filter_selectivity
    - k
    - m
    - ef_search
    - mean_recall_attidx0
    - QPS_attidx0
    - mean_recall_attidx1
    - QPS_attidx1
    """
    root_results = '/home/abylay/ann-benchmarks-HQ/results'
    
    # Load data for both att_idx values
    data_dict = {}
    for att_idx in [0, 1]:
        csv_path = f"{root_results}/MoRe_UPD_{dataset_size}_attidx_{att_idx}/all_results_hnsw.csv"
        try:
            df = pd.read_csv(csv_path)
            data_dict[att_idx] = compute_averages(df)
            print(f"Loaded data for att_idx={att_idx}: {len(data_dict[att_idx])} rows")
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            data_dict[att_idx] = None
    
    if data_dict[0] is None and data_dict[1] is None:
        print("No data available for either att_idx")
        return None
    
    # Filter for specific query type and algorithm
    filtered_data = {}
    for att_idx in [0, 1]:
        if data_dict[att_idx] is not None:
            filtered = data_dict[att_idx][
                (data_dict[att_idx]['query_type'] == query_type) &
                (data_dict[att_idx]['algorithm'] == algorithm)
            ]
            filtered_data[att_idx] = filtered
            print(f"Filtered data for att_idx={att_idx} ({query_type}, {algorithm}): {len(filtered)} rows")
        else:
            filtered_data[att_idx] = pd.DataFrame()
    
    # Group by filter_selectivity, k, m, ef_search and compute means
    comparison_data = []
    
    # Get all unique combinations from both datasets
    all_combinations = set()
    for att_idx in [0, 1]:
        if len(filtered_data[att_idx]) > 0:
            for _, row in filtered_data[att_idx].iterrows():
                key = (row['filter_selectivity'], row['k'], row['m'], row['ef_search'])
                all_combinations.add(key)
    
    # Process each combination
    for sel, k, m, ef in sorted(all_combinations):
        row_data = {
            'filter_selectivity': sel,
            'k': k,
            'm': m,
            'ef_search': ef
        }
        
        # Get data for att_idx=0
        if len(filtered_data[0]) > 0:
            data_0 = filtered_data[0][
                (filtered_data[0]['filter_selectivity'] == sel) &
                (filtered_data[0]['k'] == k) &
                (filtered_data[0]['m'] == m) &
                (filtered_data[0]['ef_search'] == ef)
            ]
            if len(data_0) > 0:
                row_data['mean_recall_attidx0'] = data_0['recall'].mean()
                row_data['QPS_attidx0'] = data_0['throughput'].mean()
            else:
                row_data['mean_recall_attidx0'] = np.nan
                row_data['QPS_attidx0'] = np.nan
        else:
            row_data['mean_recall_attidx0'] = np.nan
            row_data['QPS_attidx0'] = np.nan
        
        # Get data for att_idx=1
        if len(filtered_data[1]) > 0:
            data_1 = filtered_data[1][
                (filtered_data[1]['filter_selectivity'] == sel) &
                (filtered_data[1]['k'] == k) &
                (filtered_data[1]['m'] == m) &
                (filtered_data[1]['ef_search'] == ef)
            ]
            if len(data_1) > 0:
                row_data['mean_recall_attidx1'] = data_1['recall'].mean()
                row_data['QPS_attidx1'] = data_1['throughput'].mean()
            else:
                row_data['mean_recall_attidx1'] = np.nan
                row_data['QPS_attidx1'] = np.nan
        else:
            row_data['mean_recall_attidx1'] = np.nan
            row_data['QPS_attidx1'] = np.nan
        
        comparison_data.append(row_data)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Sort by filter_selectivity, k, m, ef_search
    df_comparison = df_comparison.sort_values(['filter_selectivity', 'k', 'm', 'ef_search'])
    
    return df_comparison

def main():
    """Main function to analyze recall comparison"""
    dataset_sizes = ['small', 'medium', 'large']
    algorithms = ['pgvector', 'milvus-hnsw']
    query_types = ['movies', 'reviews']
    
    root_results = '/home/abylay/ann-benchmarks-HQ/results'
    output_dir = f"{root_results}/MoRe_UPD_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_size in dataset_sizes:
        for algorithm in algorithms:
            for query_type in query_types:
                print(f"\n{'='*80}")
                print(f"Analyzing: {dataset_size}, {algorithm}, {query_type}")
                print(f"{'='*80}")
                
                df_comparison = analyze_recall_comparison(dataset_size, algorithm, query_type)
                
                if df_comparison is not None and len(df_comparison) > 0:
                    # Print the dataframe
                    pd.set_option('display.max_rows', None)
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    pd.set_option('display.max_colwidth', None)
                    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if not pd.isna(x) else 'NaN')
                    
                    print(f"\nComparison DataFrame ({len(df_comparison)} rows):")
                    print(df_comparison.to_string(index=False))
                    
                    # Save to CSV
                    output_file = f"{output_dir}/recall_comparison_{dataset_size}_{algorithm}_{query_type}.csv"
                    df_comparison.to_csv(output_file, index=False)
                    print(f"\nSaved to: {output_file}")
                    
                    # Print summary statistics
                    print(f"\nSummary Statistics:")
                    print(f"  Rows with att_idx=0 data: {df_comparison['mean_recall_attidx0'].notna().sum()}")
                    print(f"  Rows with att_idx=1 data: {df_comparison['mean_recall_attidx1'].notna().sum()}")
                    print(f"  Rows with both: {(df_comparison['mean_recall_attidx0'].notna() & df_comparison['mean_recall_attidx1'].notna()).sum()}")
                    
                    # Analyze recall differences
                    both_available = df_comparison[
                        df_comparison['mean_recall_attidx0'].notna() & 
                        df_comparison['mean_recall_attidx1'].notna()
                    ]
                    if len(both_available) > 0:
                        print(f"\nRecall Analysis (where both att_idx values available):")
                        print(f"  Mean recall att_idx=0: {both_available['mean_recall_attidx0'].mean():.4f}")
                        print(f"  Mean recall att_idx=1: {both_available['mean_recall_attidx1'].mean():.4f}")
                        print(f"  Cases where att_idx=1 recall = 1.0: {(both_available['mean_recall_attidx1'] == 1.0).sum()}")
                        print(f"  Cases where att_idx=1 recall < 1.0: {(both_available['mean_recall_attidx1'] < 1.0).sum()}")
                        
                        # Check high selectivity cases (< 0.1)
                        high_sel = both_available[both_available['filter_selectivity'] < 0.1]
                        if len(high_sel) > 0:
                            print(f"\nHigh Selectivity (< 0.1) Analysis:")
                            print(f"  Total rows: {len(high_sel)}")
                            print(f"  Mean recall att_idx=0: {high_sel['mean_recall_attidx0'].mean():.4f}")
                            print(f"  Mean recall att_idx=1: {high_sel['mean_recall_attidx1'].mean():.4f}")
                            print(f"  Cases where att_idx=1 recall = 1.0: {(high_sel['mean_recall_attidx1'] == 1.0).sum()}")
                            print(f"  Cases where att_idx=1 recall < 1.0: {(high_sel['mean_recall_attidx1'] < 1.0).sum()}")
                            
                            # Show cases where recall < 1.0 at high selectivity
                            low_recall_high_sel = high_sel[high_sel['mean_recall_attidx1'] < 1.0]
                            if len(low_recall_high_sel) > 0:
                                print(f"\n  Cases with recall < 1.0 at high selectivity:")
                                print(low_recall_high_sel[['filter_selectivity', 'k', 'm', 'ef_search', 'mean_recall_attidx0', 'mean_recall_attidx1']].to_string(index=False))
                else:
                    print("No comparison data available")
                
                print(f"\n{'-'*80}\n")

if __name__ == "__main__":
    main()

