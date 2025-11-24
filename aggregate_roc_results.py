#!/usr/bin/env python3

import pandas as pd
import glob
import re
import argparse
from pathlib import Path

def extract_language_from_filename(filename):
    """Extract target language from filename like aya-23-8B_kgw_seed0_de_length_analysis_roc.csv"""
    match = re.search(r'_([a-z]{2})_length_analysis_roc\.csv$', filename)
    if match:
        return match.group(1)
    return None

def aggregate_roc_results(pattern, output_file=None):
    """
    Aggregate ROC results from multiple language CSV files into one comprehensive table.

    Args:
        pattern: Glob pattern to match CSV files (e.g., "*aya-23-8B_kgw_seed0_*_length_analysis_roc.csv")
        output_file: Output CSV file path (optional)

    Returns:
        pandas.DataFrame: Aggregated results table
    """

    # Find all matching files
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No files found matching pattern: {pattern}")
        return None

    print(f"Found {len(csv_files)} ROC CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    print()

    # Collect data from all files
    all_data = []

    for csv_file in csv_files:
        # Extract language from filename
        lang = extract_language_from_filename(csv_file)
        if not lang:
            print(f"Warning: Could not extract language from filename: {csv_file}")
            continue

        # Read the CSV file
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        # Add language column and append to collection
        df['Language'] = lang.upper()
        all_data.append(df)

    if not all_data:
        print("No valid data found in any files.")
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    # Reorder columns to put Language first
    columns = ['Language', 'Length_Category', 'AUC', 'TPR_at_FPR_0.01', 'TPR_at_FPR_0.1']
    combined_df = combined_df[columns]

    # Sort by language and then by length category order
    length_order = {'Short': 1, 'Medium': 2, 'Long': 3}
    combined_df['sort_key'] = combined_df['Length_Category'].map(length_order)
    combined_df = combined_df.sort_values(['Language', 'sort_key'])
    combined_df = combined_df.drop('sort_key', axis=1)

    # Reset index
    combined_df = combined_df.reset_index(drop=True)

    # Print the aggregated table
    print("=== AGGREGATED ROC RESULTS FOR ALL LANGUAGES ===")
    print(combined_df.to_string(index=False, float_format='%.3f'))
    print()

    # Create pivot tables for easier analysis
    print("=== AUC SCORES BY LANGUAGE AND LENGTH ===")
    auc_pivot = combined_df.pivot(index='Language', columns='Length_Category', values='AUC')
    auc_pivot = auc_pivot[['Short', 'Medium', 'Long']]  # Ensure correct order
    print(auc_pivot.round(3).to_string())
    print()

    print("=== TPR@FPR=0.01 BY LANGUAGE AND LENGTH ===")
    tpr_001_pivot = combined_df.pivot(index='Language', columns='Length_Category', values='TPR_at_FPR_0.01')
    tpr_001_pivot = tpr_001_pivot[['Short', 'Medium', 'Long']]
    print(tpr_001_pivot.round(3).to_string())
    print()

    print("=== TPR@FPR=0.1 BY LANGUAGE AND LENGTH ===")
    tpr_01_pivot = combined_df.pivot(index='Language', columns='Length_Category', values='TPR_at_FPR_0.1')
    tpr_01_pivot = tpr_01_pivot[['Short', 'Medium', 'Long']]
    print(tpr_01_pivot.round(3).to_string())
    print()

    # Save to file if specified
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"✅ Aggregated results saved to: {output_file}")

        # Save pivot tables as well
        base_name = output_file.replace('.csv', '')

        auc_pivot.to_csv(f"{base_name}_auc_pivot.csv")
        tpr_001_pivot.to_csv(f"{base_name}_tpr_001_pivot.csv")
        tpr_01_pivot.to_csv(f"{base_name}_tpr_01_pivot.csv")

        print(f"✅ Pivot tables saved:")
        print(f"   - AUC: {base_name}_auc_pivot.csv")
        print(f"   - TPR@0.01: {base_name}_tpr_001_pivot.csv")
        print(f"   - TPR@0.1: {base_name}_tpr_01_pivot.csv")

    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Aggregate ROC results from multiple language CSV files")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*aya-23-8B_kgw_seed0_*_length_analysis_roc.csv",
        help="Glob pattern to match CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_length_analysis",
        help="Directory containing the CSV files"
    )

    args = parser.parse_args()

    # Construct full pattern path
    if args.results_dir:
        pattern = f"{args.results_dir}/{args.pattern}"
    else:
        pattern = args.pattern

    # Set default output file if not specified
    if not args.output and args.results_dir:
        args.output = f"{args.results_dir}/aggregated_roc_results.csv"

    print(f"=== AGGREGATING ROC RESULTS ===")
    print(f"Pattern: {pattern}")
    print(f"Output: {args.output or 'Console only'}")
    print()

    # Aggregate the results
    df = aggregate_roc_results(pattern, args.output)

    if df is not None:
        print(f"\n🎉 Successfully aggregated {len(df)} rows from {len(df['Language'].unique())} languages")
        print(f"Languages included: {', '.join(sorted(df['Language'].unique()))}")
    else:
        print("❌ No results to aggregate")

if __name__ == "__main__":
    main()