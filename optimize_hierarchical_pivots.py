#!/usr/bin/env python3
"""
Hierarchical STEAM Pivot Language Optimization Script

Uses hierarchical cluster-aware optimization that:
1. First selects optimal clusters
2. Then selects optimal language representatives within each cluster

This is more efficient than using all languages from selected clusters.

Usage:
    python3 optimize_hierarchical_pivots.py --target_lang fra --output_file results.json
"""

import argparse
import json
import sys
from pathlib import Path

from hierarchical_cluster_optimizer import HierarchicalComparator


def main():
    parser = argparse.ArgumentParser(
        description="Optimize STEAM pivot languages using hierarchical genetic distance strategies"
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
        "--langs_per_cluster",
        type=int,
        default=2,
        help="Maximum languages per cluster (default: 2)"
    )
    parser.add_argument(
        "--cluster_file",
        default="clusters_genetic.csv",
        help="Path to genetic cluster file (default: clusters_genetic.csv)"
    )

    args = parser.parse_args()

    print(f"🧬 HIERARCHICAL STEAM Pivot Language Optimization")
    print(f"🎯 Target language: {args.target_lang}")
    print(f"📊 Evaluations per strategy: {args.n_calls}")
    print(f"📁 Output file: {args.output_file}")
    print(f"🔗 Max clusters: {args.max_clusters}")
    print(f"👥 Languages per cluster: {args.langs_per_cluster}")
    print("=" * 70)

    try:
        # Initialize hierarchical comparator
        comparator = HierarchicalComparator(cluster_file=args.cluster_file)

        # Run hierarchical optimization with all 4 strategies
        print(f"🚀 Running hierarchical optimization for {args.target_lang}...")
        results = comparator.compare_all_strategies(
            target_lang=args.target_lang,
            n_calls=args.n_calls,
            max_clusters=args.max_clusters,
            langs_per_cluster=args.langs_per_cluster
        )

        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        comparator.save_results(results, args.output_file)

        # Print summary
        best_strategy = results['comparison']['best_strategy']
        best_score = results['comparison']['best_score']
        best_clusters = results[best_strategy]['best_clusters']
        best_languages = results[best_strategy]['best_languages']
        language_selection = results[best_strategy]['language_selection_by_cluster']

        print("\n" + "=" * 70)
        print("🏆 HIERARCHICAL OPTIMIZATION SUMMARY")
        print("=" * 70)
        print(f"🥇 Best strategy: {best_strategy.replace('_', ' ').title()}")
        print(f"📈 Best score: {best_score:.4f}")
        print(f"🎪 Optimal clusters: {best_clusters}")
        print(f"🗣️  Selected pivot languages: {best_languages}")

        print(f"\n📋 Language selection by cluster:")
        for cluster_id, langs in language_selection.items():
            print(f"   Cluster {cluster_id}: {langs}")

        print(f"\n💾 Results saved to: {args.output_file}")
        print("✅ Hierarchical optimization completed successfully!")

    except Exception as e:
        print(f"❌ Error during hierarchical optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()