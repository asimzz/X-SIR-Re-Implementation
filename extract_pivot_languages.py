#!/usr/bin/env python3
"""
Extract Pivot Languages from Optimization Results

This script extracts the best_languages for each strategy and target language
from both simple and hierarchical optimization results, making it easy to
use in translation experiments.

Usage:
    python3 extract_pivot_languages.py --output_csv pivot_languages.csv
"""

import json
import pandas as pd
import argparse
from pathlib import Path
import glob


def extract_from_simple_results(results_dir="steam_optimization_results"):
    """Extract pivot languages from simple optimization results"""

    simple_data = []
    results_pattern = f"{results_dir}/steam_optimization_*.json"

    for result_file in glob.glob(results_pattern):
        target_lang = Path(result_file).stem.replace("steam_optimization_", "")

        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Extract best languages for each strategy
            strategies = ['max_distance', 'min_distance', 'diversity', 'gp_learning']

            for strategy in strategies:
                if strategy in data and 'best_evaluation' in data[strategy]:
                    pivot_languages = data[strategy]['best_evaluation']['pivot_languages']
                    best_clusters = data[strategy]['best_clusters']
                    score = data[strategy]['best_score']

                    simple_data.append({
                        'target_language': target_lang,
                        'optimization_type': 'simple',
                        'strategy': strategy,
                        'best_languages': pivot_languages,
                        'best_clusters': best_clusters,
                        'score': score,
                        'num_languages': len(pivot_languages) if pivot_languages else 0
                    })

            # Add overall winner
            if 'comparison' in data:
                winner_strategy = data['comparison']['best_strategy']
                winner_score = data['comparison']['best_score']
                winner_languages = data[winner_strategy]['best_evaluation']['pivot_languages']
                winner_clusters = data[winner_strategy]['best_clusters']

                simple_data.append({
                    'target_language': target_lang,
                    'optimization_type': 'simple',
                    'strategy': 'winner',
                    'best_languages': winner_languages,
                    'best_clusters': winner_clusters,
                    'score': winner_score,
                    'num_languages': len(winner_languages) if winner_languages else 0
                })

        except Exception as e:
            print(f"Error processing {result_file}: {e}")

    return simple_data


def extract_from_hierarchical_results(results_dir="hierarchical_optimization_results"):
    """Extract pivot languages from hierarchical optimization results"""

    hierarchical_data = []
    results_pattern = f"{results_dir}/hierarchical_optimization_*.json"

    for result_file in glob.glob(results_pattern):
        target_lang = Path(result_file).stem.replace("hierarchical_optimization_", "")

        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Extract best languages for each strategy
            strategies = ['max_distance', 'min_distance', 'diversity', 'gp_learning']

            for strategy in strategies:
                if strategy in data:
                    best_languages = data[strategy].get('best_languages', [])
                    best_clusters = data[strategy].get('best_clusters', [])
                    language_selection = data[strategy].get('language_selection_by_cluster', {})
                    score = data[strategy].get('best_score', 0.0)

                    hierarchical_data.append({
                        'target_language': target_lang,
                        'optimization_type': 'hierarchical',
                        'strategy': strategy,
                        'best_languages': best_languages,
                        'best_clusters': best_clusters,
                        'language_selection_by_cluster': language_selection,
                        'score': score,
                        'num_languages': len(best_languages) if best_languages else 0
                    })

            # Add overall winner
            if 'comparison' in data:
                winner_strategy = data['comparison']['best_strategy']
                winner_score = data['comparison']['best_score']
                winner_data = data[winner_strategy]

                hierarchical_data.append({
                    'target_language': target_lang,
                    'optimization_type': 'hierarchical',
                    'strategy': 'winner',
                    'best_languages': winner_data.get('best_languages', []),
                    'best_clusters': winner_data.get('best_clusters', []),
                    'language_selection_by_cluster': winner_data.get('language_selection_by_cluster', {}),
                    'score': winner_score,
                    'num_languages': len(winner_data.get('best_languages', []))
                })

        except Exception as e:
            print(f"Error processing {result_file}: {e}")

    return hierarchical_data


def create_summary_tables(all_data):
    """Create summary tables for easy analysis"""

    df = pd.DataFrame(all_data)

    # Summary by target language and optimization type
    summary_by_target = df.groupby(['target_language', 'optimization_type']).agg({
        'num_languages': 'mean',
        'score': 'max'
    }).round(4)

    # Winner strategies comparison
    winners_df = df[df['strategy'] == 'winner'].copy()
    winners_comparison = winners_df.pivot_table(
        index='target_language',
        columns='optimization_type',
        values=['score', 'num_languages'],
        aggfunc='first'
    )

    return df, summary_by_target, winners_comparison


def save_results(df, summary_by_target, winners_comparison, output_dir="pivot_language_extraction"):
    """Save all results to files"""

    Path(output_dir).mkdir(exist_ok=True)

    # Save detailed results
    df.to_csv(f"{output_dir}/all_pivot_languages.csv", index=False)

    # Save summary tables
    summary_by_target.to_csv(f"{output_dir}/summary_by_target.csv")
    winners_comparison.to_csv(f"{output_dir}/winners_comparison.csv")

    # Save strategy-specific files for easy translation experiments
    strategies = ['max_distance', 'min_distance', 'diversity', 'gp_learning', 'winner']

    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy].copy()

        # Create pivot table: target_language x optimization_type
        pivot_table = strategy_df.pivot_table(
            index='target_language',
            columns='optimization_type',
            values='best_languages',
            aggfunc='first'
        )

        # Save as CSV
        pivot_table.to_csv(f"{output_dir}/{strategy}_pivot_languages.csv")

        # Create a clean format for translation scripts
        for opt_type in ['simple', 'hierarchical']:
            clean_df = strategy_df[strategy_df['optimization_type'] == opt_type][
                ['target_language', 'best_languages', 'score']
            ].copy()
            clean_df.to_csv(f"{output_dir}/{strategy}_{opt_type}_clean.csv", index=False)

    print(f"Results saved to {output_dir}/")
    return output_dir


def print_summary(df, winners_comparison):
    """Print summary statistics"""

    print("\n" + "="*70)
    print("PIVOT LANGUAGE EXTRACTION SUMMARY")
    print("="*70)

    total_targets = df['target_language'].nunique()
    print(f"📊 Total target languages processed: {total_targets}")

    # Strategy distribution
    strategy_counts = df.groupby(['optimization_type', 'strategy']).size().unstack(fill_value=0)
    print(f"\n📋 Strategies extracted:")
    print(strategy_counts)

    # Average number of pivot languages per strategy
    avg_languages = df.groupby(['optimization_type', 'strategy'])['num_languages'].mean()
    print(f"\n📈 Average pivot languages per strategy:")
    for (opt_type, strategy), avg in avg_languages.items():
        print(f"  {opt_type:12} {strategy:12}: {avg:.1f} languages")

    # Winner comparison
    print(f"\n🏆 Winner Strategy Distribution:")
    winner_df = df[df['strategy'] == 'winner']

    for opt_type in ['simple', 'hierarchical']:
        type_winners = winner_df[winner_df['optimization_type'] == opt_type]
        if not type_winners.empty:
            avg_languages = type_winners['num_languages'].mean()
            total_targets = len(type_winners)
            print(f"  {opt_type:12}: {total_targets} targets, avg {avg_languages:.1f} pivot languages")

    print(f"\n💾 Files created for translation experiments:")
    print(f"  - all_pivot_languages.csv: Complete dataset")
    print(f"  - winner_simple_clean.csv: Winners from simple optimization")
    print(f"  - winner_hierarchical_clean.csv: Winners from hierarchical optimization")
    print(f"  - {strategy}_pivot_languages.csv: Per-strategy comparisons")


def main():
    parser = argparse.ArgumentParser(
        description="Extract pivot languages from optimization results"
    )
    parser.add_argument(
        "--simple_results_dir",
        default="steam_optimization_results",
        help="Directory with simple optimization results"
    )
    parser.add_argument(
        "--hierarchical_results_dir",
        default="hierarchical_optimization_results",
        help="Directory with hierarchical optimization results"
    )
    parser.add_argument(
        "--output_dir",
        default="pivot_language_extraction",
        help="Output directory for extracted results"
    )

    args = parser.parse_args()

    print("🔍 Extracting pivot languages from optimization results...")

    # Extract from both result types
    simple_data = extract_from_simple_results(args.simple_results_dir)
    hierarchical_data = extract_from_hierarchical_results(args.hierarchical_results_dir)

    print(f"✅ Extracted {len(simple_data)} simple optimization entries")
    print(f"✅ Extracted {len(hierarchical_data)} hierarchical optimization entries")

    # Combine data
    all_data = simple_data + hierarchical_data

    # Create summary tables
    df, summary_by_target, winners_comparison = create_summary_tables(all_data)

    # Save results
    output_dir = save_results(df, summary_by_target, winners_comparison, args.output_dir)

    # Print summary
    print_summary(df, winners_comparison)

    print(f"\n🎉 Pivot language extraction complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n🚀 Ready for translation experiments!")


if __name__ == "__main__":
    main()