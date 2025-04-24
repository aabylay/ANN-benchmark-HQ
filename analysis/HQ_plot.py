import pandas as pd

result_file_name = "all_results.csv"
result_df = pd.read_csv(f'results/{result_file_name}')

"""
def print_aggregations(result_df):
    # Ensure result_df is not empty
    if result_df.empty:
        print("DataFrame is empty. No aggregations to display.")
        return

    # Extract idx_number columns (those ending with '_recall' or '_runtime')
    recall_cols = [col for col in result_df.columns if col.endswith('_recall')]
    idx_numbers = [col.replace('_recall', '') for col in recall_cols]

    # 1) idx-k | recall | runtime
    print("\n1) Average Recall and Runtime by idx and k:")
    print("-" * 50)
    for idx in idx_numbers:
        recall_col = f"{idx}_recall"
        runtime_col = f"{idx}_runtime"
        
        # Group by k and calculate mean for recall and runtime
        agg_df = result_df.groupby('k').agg({
            recall_col: 'mean',
            runtime_col: 'mean'
        }).reset_index()
        
        print(f"\nidx: {idx}")
        print(agg_df.rename(columns={
            recall_col: 'recall',
            runtime_col: 'runtime'
        })[['k', 'recall', 'runtime']].to_string(index=False))

    # 2) idx-filter | recall | runtime
    print("\n2) Average Recall and Runtime by idx and filter:")
    print("-" * 50)
    for idx in idx_numbers:
        recall_col = f"{idx}_recall"
        runtime_col = f"{idx}_runtime"
        
        # Group by filter and calculate mean for recall and runtime
        agg_df = result_df.groupby('filter').agg({
            recall_col: 'mean',
            runtime_col: 'mean'
        }).reset_index()
        
        print(f"\nidx: {idx}")
        print(agg_df.rename(columns={
            recall_col: 'recall',
            runtime_col: 'runtime'
        })[['filter', 'recall', 'runtime']].to_string(index=False))

# Example usage (assuming result_df is already created from the previous script)
# Replace this with your actual result_df
def main():
    # Load or generate result_df (for demonstration, assuming it's available)
    # result_df = pd.read_csv('combined_results.csv')  # If saved previously
    try:
        print_aggregations(result_df)
    except NameError:
        print("Error: result_df not found. Please ensure result_df is generated from the previous script.")
"""
def main():
    print(result_df.tail())

if __name__ == "__main__":
    main()