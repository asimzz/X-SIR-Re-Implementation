#!/usr/bin/env python3
"""
Hierarchical Cluster-Aware Genetic Distance Optimizer

This optimizer implements a two-level optimization approach:
1. First level: Select optimal clusters based on genetic distance strategy
2. Second level: Select optimal language representatives within each cluster

This uses clusters more intelligently than the simple approach by finding
the best language representatives from each cluster rather than using all languages.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import json
from dataclasses import dataclass

from skopt import gp_minimize
from skopt.space import Categorical, Integer

from genetic_distance_evaluator import GeneticDistanceEvaluator, GeneticDistanceResult


@dataclass
class HierarchicalResult:
    """Result of hierarchical cluster optimization"""
    cluster_selection: List[int]
    language_selection: Dict[int, List[str]]  # cluster_id -> selected languages
    all_selected_languages: List[str]
    genetic_distances: Dict[str, float]
    distance_features: Dict[str, float]
    strategy_score: float
    strategy_type: str
    cluster_scores: Dict[int, float]  # Individual cluster genetic scores


class HierarchicalClusterOptimizer:
    """
    Two-level hierarchical optimizer:
    1. Cluster-level optimization using genetic distance strategies
    2. Language-level optimization within selected clusters
    """

    def __init__(self, cluster_file: str = "clusters_genetic.csv", strategy: str = "max_distance"):
        """
        Initialize hierarchical optimizer.

        Args:
            cluster_file: Path to genetic cluster CSV file
            strategy: One of ['max_distance', 'min_distance', 'diversity', 'gp_learning']
        """
        self.strategy = strategy
        self.evaluator = GeneticDistanceEvaluator(cluster_file)
        self.available_clusters = list(self.evaluator.cluster_to_languages.keys())

        # Optimization tracking
        self.evaluation_history = []
        self.current_target_lang = None

        print(f"Hierarchical Cluster Optimizer initialized with '{strategy}' strategy")
        print(f"Available clusters: {self.available_clusters}")
        print(f"Total languages: {sum(len(langs) for langs in self.evaluator.cluster_to_languages.values())}")

    def get_cluster_genetic_score(self, cluster_id: int, target_lang: str) -> Tuple[float, List[str]]:
        """
        Evaluate a single cluster's genetic distance characteristics to target.
        Returns the cluster's genetic score and best representative languages.
        """
        if cluster_id not in self.evaluator.cluster_to_languages:
            return 0.0, []

        cluster_langs = self.evaluator.cluster_to_languages[cluster_id]
        if target_lang in cluster_langs:
            cluster_langs = [lang for lang in cluster_langs if lang != target_lang]

        if not cluster_langs:
            return 0.0, []

        # Calculate genetic distances for all languages in cluster
        lang_distances = []
        for lang in cluster_langs:
            distance = self.evaluator.get_genetic_distance(target_lang, lang)
            lang_distances.append((lang, distance))

        # Sort by genetic distance based on strategy
        if self.strategy == "max_distance":
            # Want most distant languages from this cluster
            lang_distances.sort(key=lambda x: x[1], reverse=True)
            cluster_score = np.mean([dist for _, dist in lang_distances])
        elif self.strategy == "min_distance":
            # Want closest languages from this cluster
            lang_distances.sort(key=lambda x: x[1])
            cluster_score = 1.0 - np.mean([dist for _, dist in lang_distances])
        elif self.strategy == "diversity":
            # Want diverse distance range within cluster
            distances = [dist for _, dist in lang_distances]
            cluster_score = np.std(distances) if len(distances) > 1 else 0.0
        else:  # gp_learning
            # Rich features for GP to learn from
            distances = [dist for _, dist in lang_distances]
            cluster_score = np.mean([
                np.mean(distances),
                np.max(distances) if distances else 0,
                np.std(distances) if len(distances) > 1 else 0,
                len(distances)
            ])

        # Select top representatives from this cluster (up to 3)
        top_representatives = [lang for lang, _ in lang_distances[:3]]

        return cluster_score, top_representatives

    def evaluate_cluster_combination(self, cluster_ids: List[int], target_lang: str,
                                   langs_per_cluster: int = 2) -> HierarchicalResult:
        """
        Evaluate a combination of clusters with optimal language selection within each.

        Args:
            cluster_ids: List of cluster IDs to evaluate
            target_lang: Target language
            langs_per_cluster: Max languages to select per cluster
        """
        if not cluster_ids:
            return HierarchicalResult(
                cluster_selection=[],
                language_selection={},
                all_selected_languages=[],
                genetic_distances={},
                distance_features={"mean_distance": 0.0, "max_distance": 0.0,
                                 "min_distance": 0.0, "distance_diversity": 0.0},
                strategy_score=0.0,
                strategy_type=self.strategy,
                cluster_scores={}
            )

        # Step 1: Evaluate each cluster and get representatives
        cluster_scores = {}
        language_selection = {}
        all_selected_languages = []

        for cluster_id in cluster_ids:
            score, representatives = self.get_cluster_genetic_score(cluster_id, target_lang)
            cluster_scores[cluster_id] = score

            # Select top languages from this cluster
            selected = representatives[:langs_per_cluster]
            language_selection[cluster_id] = selected
            all_selected_languages.extend(selected)

        # Step 2: Calculate overall genetic distance features
        if not all_selected_languages:
            return HierarchicalResult(
                cluster_selection=cluster_ids,
                language_selection=language_selection,
                all_selected_languages=[],
                genetic_distances={},
                distance_features={"mean_distance": 0.0, "max_distance": 0.0,
                                 "min_distance": 0.0, "distance_diversity": 0.0},
                strategy_score=0.0,
                strategy_type=self.strategy,
                cluster_scores=cluster_scores
            )

        # Calculate genetic distances for all selected languages
        genetic_distances = {}
        distances = []
        for lang in all_selected_languages:
            dist = self.evaluator.get_genetic_distance(target_lang, lang)
            genetic_distances[lang] = dist
            distances.append(dist)

        distances = np.array(distances)

        # Distance features
        distance_features = {
            "mean_distance": np.mean(distances),
            "max_distance": np.max(distances),
            "min_distance": np.min(distances),
            "distance_diversity": np.std(distances) if len(distances) > 1 else 0.0,
            "distance_range": np.max(distances) - np.min(distances) if len(distances) > 1 else 0.0,
            "num_languages": len(distances),
            "num_clusters": len(cluster_ids)
        }

        # Step 3: Calculate strategy-specific score
        if self.strategy == "max_distance":
            strategy_score = distance_features["mean_distance"]
        elif self.strategy == "min_distance":
            strategy_score = 1.0 - distance_features["mean_distance"]
        elif self.strategy == "diversity":
            strategy_score = 0.6 * distance_features["distance_diversity"] + 0.4 * distance_features["distance_range"]
        else:  # gp_learning
            # Rich composite score
            strategy_score = np.mean([
                distance_features["mean_distance"],
                distance_features["distance_diversity"],
                distance_features["distance_range"] / 2.0,  # Normalize range
                len(cluster_ids) / 5.0,  # Cluster diversity bonus
                len(all_selected_languages) / 10.0  # Language count bonus
            ])

        return HierarchicalResult(
            cluster_selection=cluster_ids,
            language_selection=language_selection,
            all_selected_languages=all_selected_languages,
            genetic_distances=genetic_distances,
            distance_features=distance_features,
            strategy_score=strategy_score,
            strategy_type=self.strategy,
            cluster_scores=cluster_scores
        )

    def create_hierarchical_search_space(self, max_clusters: int = 3) -> List:
        """Create search space for hierarchical cluster + language selection"""
        # Use limited clusters for manageable search
        limited_clusters = self.available_clusters[:min(15, len(self.available_clusters))]

        dimensions = []
        for i in range(max_clusters):
            # Each position can select a cluster or -1 for "none"
            choices = [-1] + limited_clusters
            dimensions.append(Categorical(choices, name=f'cluster_pos_{i}'))

        return dimensions

    def decode_search_point(self, x: List[int]) -> List[int]:
        """Convert search space point to cluster list"""
        selected_clusters = []
        for cluster_id in x:
            if cluster_id != -1 and cluster_id not in selected_clusters:
                selected_clusters.append(cluster_id)
        return sorted(selected_clusters)

    def objective_function(self, x: List[int], max_clusters: int = 3, langs_per_cluster: int = 2) -> float:
        """
        Objective function for hierarchical optimization.
        Returns NEGATIVE score since skopt minimizes.
        """
        clusters = self.decode_search_point(x)

        if len(clusters) == 0 or len(clusters) > max_clusters:
            return 1000.0  # Large penalty

        result = self.evaluate_cluster_combination(clusters, self.current_target_lang, langs_per_cluster)

        # Track evaluation
        evaluation_record = {
            "iteration": len(self.evaluation_history) + 1,
            "clusters": clusters,
            "selected_languages": result.all_selected_languages,
            "language_selection": result.language_selection,
            "score": result.strategy_score,
            "strategy": self.strategy,
            "cluster_scores": result.cluster_scores,
            "distance_features": result.distance_features
        }
        self.evaluation_history.append(evaluation_record)

        print(f"  Eval {len(self.evaluation_history)}: Clusters {clusters} -> Score: {result.strategy_score:.4f}")
        print(f"    Languages: {result.all_selected_languages}")
        print(f"    Mean distance: {result.distance_features['mean_distance']:.3f}")

        return -result.strategy_score  # Negative for minimization

    def optimize_hierarchical(self, target_lang: str, n_calls: int = 25, max_clusters: int = 3,
                            langs_per_cluster: int = 2, random_state: int = 42) -> Dict:
        """
        Run hierarchical Bayesian optimization.

        Args:
            target_lang: Target language (e.g., 'fra' for French)
            n_calls: Total number of evaluations
            max_clusters: Maximum clusters to select
            langs_per_cluster: Maximum languages per cluster
            random_state: Random seed
        """
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL GENETIC DISTANCE OPTIMIZATION: {self.strategy.upper()}")
        print(f"{'='*70}")
        print(f"Target language: {target_lang}")
        print(f"Strategy: {self.strategy}")
        print(f"Max clusters: {max_clusters}")
        print(f"Languages per cluster: {langs_per_cluster}")
        print(f"Evaluations: {n_calls}")

        # Set context
        self.current_target_lang = target_lang
        self.evaluation_history = []

        # Create search space
        dimensions = self.create_hierarchical_search_space(max_clusters)

        # Create objective function
        def objective(x):
            return self.objective_function(x, max_clusters, langs_per_cluster)

        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=max(5, n_calls // 5),
            acq_func='EI',
            random_state=random_state,
            verbose=False
        )

        # Extract best result
        best_x = result.x
        best_clusters = self.decode_search_point(best_x)
        best_score = -result.fun

        # Get detailed result for best combination
        best_hierarchical_result = self.evaluate_cluster_combination(
            best_clusters, target_lang, langs_per_cluster
        )

        optimization_result = {
            'strategy': self.strategy,
            'target_language': target_lang,
            'best_clusters': best_clusters,
            'best_languages': best_hierarchical_result.all_selected_languages,
            'language_selection_by_cluster': best_hierarchical_result.language_selection,
            'cluster_scores': best_hierarchical_result.cluster_scores,
            'best_score': best_score,
            'total_evaluations': len(self.evaluation_history),
            'distance_features': best_hierarchical_result.distance_features,
            'all_evaluations': self.evaluation_history,
            'hierarchical_result': best_hierarchical_result
        }

        print(f"\n=== HIERARCHICAL OPTIMIZATION COMPLETE ({self.strategy.upper()}) ===")
        print(f"Best clusters: {best_clusters}")
        print(f"Selected languages: {best_hierarchical_result.all_selected_languages}")
        print(f"Best score: {best_score:.4f}")

        print(f"\nLanguage selection by cluster:")
        for cluster_id, langs in best_hierarchical_result.language_selection.items():
            score = best_hierarchical_result.cluster_scores.get(cluster_id, 0.0)
            print(f"  Cluster {cluster_id} (score={score:.3f}): {langs}")

        return optimization_result


class HierarchicalComparator:
    """Compare all strategies using hierarchical optimization"""

    def __init__(self, cluster_file: str = "clusters_genetic.csv"):
        self.cluster_file = cluster_file

    def compare_all_strategies(self, target_lang: str, n_calls: int = 20,
                             max_clusters: int = 2, langs_per_cluster: int = 2) -> Dict:
        """Compare all four strategies using hierarchical optimization."""

        print(f"\n{'='*80}")
        print(f"HIERARCHICAL STRATEGY COMPARISON")
        print(f"{'='*80}")
        print(f"Target language: {target_lang}")
        print(f"Max clusters: {max_clusters}, Languages per cluster: {langs_per_cluster}")

        strategies = ['max_distance', 'min_distance', 'diversity', 'gp_learning']
        results = {}

        for strategy in strategies:
            print(f"\n>>> HIERARCHICAL {strategy.upper().replace('_', ' ')} STRATEGY <<<")

            optimizer = HierarchicalClusterOptimizer(self.cluster_file, strategy)
            strategy_result = optimizer.optimize_hierarchical(
                target_lang, n_calls, max_clusters, langs_per_cluster
            )
            results[strategy] = strategy_result

        # Compare results
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL STRATEGY COMPARISON RESULTS")
        print(f"{'='*60}")

        best_strategy = None
        best_score = -np.inf

        for strategy, result in results.items():
            score = result['best_score']
            clusters = result['best_clusters']
            languages = result['best_languages']

            print(f"\n{strategy.upper().replace('_', ' ')} Strategy:")
            print(f"  Best clusters: {clusters}")
            print(f"  Selected languages: {languages}")
            print(f"  Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_strategy = strategy

        print(f"\n🏆 BEST HIERARCHICAL STRATEGY: {best_strategy.upper().replace('_', ' ')}")
        print(f"📈 Best score: {best_score:.4f}")

        results['comparison'] = {
            'best_strategy': best_strategy,
            'best_score': best_score,
            'target_language': target_lang,
            'optimization_type': 'hierarchical'
        }

        return results

    def save_results(self, results: Dict, filename: str):
        """Save hierarchical comparison results to JSON"""

        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                # Convert both keys and values
                converted_dict = {}
                for key, value in obj.items():
                    # Convert key to proper type
                    if isinstance(key, np.integer):
                        converted_key = int(key)
                    elif isinstance(key, np.floating):
                        converted_key = float(key)
                    else:
                        converted_key = key

                    # Convert value recursively
                    converted_dict[converted_key] = convert_numpy_types(value)
                return converted_dict
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return convert_numpy_types(obj.__dict__)
            else:
                return obj

        # Convert to serializable format
        serializable_results = convert_numpy_types(results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nHierarchical results saved to {filename}")


if __name__ == "__main__":
    # Test hierarchical optimization for French
    comparator = HierarchicalComparator()

    results = comparator.compare_all_strategies(
        target_lang="fra",
        n_calls=15,
        max_clusters=2,
        langs_per_cluster=2
    )

    comparator.save_results(results, "hierarchical_optimization_comparison.json")

    print(f"\n🔬 Hierarchical Optimization Complete!")
    print(f"This approach selects optimal language representatives from each cluster")
    print(f"rather than using all languages, leading to more efficient pivot selection.")