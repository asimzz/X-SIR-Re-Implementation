#!/usr/bin/env python3
"""
STEAM Pivot Language Optimization Script

Optimizes pivot languages for STEAM watermark detection using genetic distance
strategies. Replaces fixed pivot languages with adaptive, linguistically-informed
selection based on target language.

Usage:
    python3 optimize_steam_pivots.py --target_lang fra --output_file results.json
"""

import argparse
import json
import sys
from pathlib import Path

from genetic_distance_optimizer import GeneticDistanceComparator


def main():
    parser = argparse.ArgumentParser(
        description="Optimize STEAM pivot languages using genetic distance strategies"
    )
    parser.add_argument(
        "--target_lang",
        required=True,
        help="Target language code (3-letter, e.g., 'fra' for French)"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--n_calls",
        type=int,
        default=20,
        help="Number of Bayesian optimization evaluations (default: 20)"
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=2,
        help="Maximum number of clusters to select (default: 2)"
    )
    parser.add_argument(
        "--cluster_file",
        default="clusters_genetic.csv",
        help="Path to genetic cluster file (default: clusters_genetic.csv)"
    )

    args = parser.parse_args()

    print(f"🧬 STEAM Pivot Language Optimization")
    print(f"🎯 Target language: {args.target_lang}")
    print(f"📊 Evaluations per strategy: {args.n_calls}")
    print(f"📁 Output file: {args.output_file}")
    print(f"🔗 Max clusters: {args.max_clusters}")
    print("=" * 60)

    try:
        # Initialize genetic distance comparator
        comparator = GeneticDistanceComparator(cluster_file=args.cluster_file)

        # Run all 4 genetic distance strategies
        print(f"🚀 Running optimization for {args.target_lang}...")
        results = comparator.compare_all_strategies(
            target_lang=args.target_lang,
            n_calls=args.n_calls,
            max_clusters=args.max_clusters
        )

        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        comparator.save_results(results, args.output_file)

        # Print summary
        best_strategy = results['comparison']['best_strategy']
        best_score = results['comparison']['best_score']
        best_clusters = results[best_strategy]['best_clusters']

        print("\n" + "=" * 60)
        print("🏆 OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"🥇 Best strategy: {best_strategy.replace('_', ' ').title()}")
        print(f"📈 Best score: {best_score:.4f}")
        print(f"🎪 Optimal clusters: {best_clusters}")

        # Extract pivot languages
        if results[best_strategy]['best_evaluation']:
            pivot_langs = results[best_strategy]['best_evaluation']['pivot_languages']
            print(f"🗣️  Optimal pivot languages: {pivot_langs[:10]}")

        print(f"💾 Results saved to: {args.output_file}")
        print("✅ Optimization completed successfully!")

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()