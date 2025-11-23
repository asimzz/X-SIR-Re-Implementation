#!/usr/bin/env python3

import pandas as pd
import glob
import os
import argparse
from pathlib import Path

def aggregate_length_analysis_results(results_dir, output_file=None):
    """
    Aggregate all length analysis CSV files into comprehensive tables.
    Creates tables for each model with languages as rows and length categories as columns.
    """

    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_dir, "*_length_analysis.csv"))

    if not csv_files:
        print(f"No length analysis CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to aggregate")

    # Read and combine all CSV files
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Extract model and method info from filename
            filename = os.path.basename(csv_file)
            # Format: {model}_{method}_seed{seed}_{lang}_length_analysis.csv
            parts = filename.replace('_length_analysis.csv', '').split('_')

            # Find seed part and reconstruct model name
            seed_idx = None
            for i, part in enumerate(parts):
                if part.startswith('seed'):
                    seed_idx = i
                    break

            if seed_idx is not None:
                model_parts = parts[:seed_idx-1]  # Everything before method
                method = parts[seed_idx-1]        # Method (e.g., 'kgw')
                seed = parts[seed_idx]            # Seed part
                lang = parts[seed_idx+1]          # Language after seed

                model = '_'.join(model_parts)

                df['Model'] = model
                df['Method'] = method
                df['Seed'] = seed

                all_data.append(df)
            else:
                print(f"Warning: Could not parse filename {filename}")

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not all_data:
        print("No data could be loaded from CSV files")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Get unique models
    models = combined_df['Model'].unique()

    # Create aggregated tables for each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model.upper()}")
        print(f"{'='*60}")

        model_data = combined_df[combined_df['Model'] == model]

        # Create pivot table with AUC scores
        auc_table = model_data.pivot_table(
            index='Language',
            columns='Length_Category',
            values='AUC',
            aggfunc='mean'  # Average across seeds if multiple
        )

        # Reorder columns to Short, Medium, Long
        column_order = ['Short', 'Medium', 'Long']
        available_cols = [col for col in column_order if col in auc_table.columns]
        auc_table = auc_table[available_cols]

        print(f"\n🎯 AUC Scores by Text Length")
        print("-" * 40)
        # Replace NaN with dash for better readability
        auc_display = auc_table.round(3).fillna('-')
        print(auc_display.to_string())

        # Create pivot table with accuracy scores
        acc_table = model_data.pivot_table(
            index='Language',
            columns='Length_Category',
            values='Accuracy_%',
            aggfunc='mean'
        )
        acc_table = acc_table[available_cols]

        print(f"\n🎯 Accuracy (%) by Text Length")
        print("-" * 40)
        # Replace NaN with dash for better readability
        acc_display = acc_table.round(1).fillna('-')
        print(acc_display.to_string())

        # Create pivot table with sample counts
        samples_table = model_data.pivot_table(
            index='Language',
            columns='Length_Category',
            values='Num_Samples',
            aggfunc='mean'
        )
        samples_table = samples_table[available_cols]

        print(f"\n📊 Sample Counts by Text Length")
        print("-" * 40)
        # Handle NaN values in sample counts
        samples_display = samples_table.fillna(0).round(0).astype(int)
        print(samples_display.to_string())

        # Save individual model table if output file specified
        if output_file:
            base_name = output_file.replace('.csv', '')

            # Save AUC table
            auc_output = f"{base_name}_{model}_AUC.csv"
            auc_table.to_csv(auc_output)
            print(f"\n💾 AUC table saved to: {auc_output}")

            # Save Accuracy table
            acc_output = f"{base_name}_{model}_Accuracy.csv"
            acc_table.to_csv(acc_output)
            print(f"💾 Accuracy table saved to: {acc_output}")

            # Save sample counts
            samples_output = f"{base_name}_{model}_Samples.csv"
            samples_table.to_csv(samples_output)
            print(f"💾 Sample counts saved to: {samples_output}")

    # Create overall summary table
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    # Average performance across all languages by model and length
    summary_auc = combined_df.groupby(['Model', 'Length_Category'])['AUC'].mean().unstack()
    summary_acc = combined_df.groupby(['Model', 'Length_Category'])['Accuracy_%'].mean().unstack()

    # Reorder columns
    if not summary_auc.empty:
        available_cols = [col for col in column_order if col in summary_auc.columns]
        summary_auc = summary_auc[available_cols]
        summary_acc = summary_acc[available_cols]

        print(f"\n🌍 Average AUC Across All Languages")
        print("-" * 45)
        print(summary_auc.round(3).to_string())

        print(f"\n🌍 Average Accuracy (%) Across All Languages")
        print("-" * 45)
        print(summary_acc.round(1).to_string())

        if output_file:
            summary_auc_file = output_file.replace('.csv', '_summary_AUC.csv')
            summary_acc_file = output_file.replace('.csv', '_summary_Accuracy.csv')
            summary_auc.to_csv(summary_auc_file)
            summary_acc.to_csv(summary_acc_file)
            print(f"\n💾 Summary tables saved to:")
            print(f"   - {summary_auc_file}")
            print(f"   - {summary_acc_file}")

    # Save the complete raw data
    if output_file:
        raw_output = output_file.replace('.csv', '_raw_data.csv')
        combined_df.to_csv(raw_output, index=False)
        print(f"💾 Raw aggregated data saved to: {raw_output}")

    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate length analysis results into comprehensive tables")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_length_analysis",
        help="Directory containing length analysis CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aggregated_length_results.csv",
        help="Base name for output files"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        return

    print(f"🔍 Aggregating length analysis results from: {args.results_dir}")
    print(f"📝 Output base name: {args.output}")

    aggregate_length_analysis_results(args.results_dir, args.output)

    print(f"\n✅ Aggregation complete!")


if __name__ == "__main__":
    main()