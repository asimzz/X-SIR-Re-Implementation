#!/usr/bin/env python3
"""
Genetic Distance-Based Bayesian Optimizer

This module implements Bayesian optimization using ONLY real genetic distance data
from URIEL, testing four different strategies for optimal pivot language selection.

Strategies:
1. Max Distance: Optimize for maximum genetic distance from target
2. Min Distance: Optimize for minimum genetic distance from target
3. Diversity: Optimize for diverse mix of genetic distances
4. GP Learning: Let Gaussian Process learn optimal distance patterns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json

from skopt import gp_minimize
from skopt.space import Categorical

from genetic_distance_evaluator import GeneticDistanceEvaluator, GeneticDistanceResult


class GeneticDistanceOptimizer:
    """
    Bayesian optimizer for language cluster selection using genetic distance strategies.

    Uses ONLY real genetic distance data from URIEL - no assumptions about
    watermark effectiveness.
    """

    def __init__(self, cluster_file: str = "clusters_genetic.csv", strategy: str = "max_distance"):
        """
        Initialize genetic distance optimizer.

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

        # Strategy mapping (language-based)
        self.strategy_functions = {
            'max_distance': self.evaluator.evaluate_max_distance_language_strategy,
            'min_distance': self.evaluator.evaluate_min_distance_language_strategy,
            'diversity': self.evaluator.evaluate_diversity_language_strategy,
            'gp_learning': self.evaluator.evaluate_gp_learning_language_strategy
        }

        if strategy not in self.strategy_functions:
            raise ValueError(f"Strategy must be one of {list(self.strategy_functions.keys())}")

        print(f"Initialized Genetic Distance Optimizer with '{strategy}' strategy")
        print(f"Available clusters: {self.available_clusters}")

    def create_search_space(self, max_languages: int = 10) -> List:
        """Create search space for individual language selection"""

        # Get all available languages from clusters
        all_languages = []
        for cluster_id, languages in self.evaluator.cluster_to_languages.items():
            all_languages.extend(languages)

        # Remove duplicates and sort
        unique_languages = sorted(list(set(all_languages)))
        print(f"Available languages for optimization: {len(unique_languages)}")

        dimensions = []
        for i in range(max_languages):
            # Each position can select a language or "none"
            choices = ["none"] + unique_languages
            dimensions.append(Categorical(choices, name=f'language_pos_{i}'))

        return dimensions

    def decode_search_point(self, x: List[str]) -> List[str]:
        """Convert search space point to language list"""
        selected_languages = []
        for language in x:
            if language != "none" and language not in selected_languages:
                selected_languages.append(language)
        return sorted(selected_languages)

    def evaluate_language_combination(self, languages: List[str], target_lang: str) -> float:
        """
        Evaluate language combination using genetic distance strategy.

        This is the objective function that Bayesian optimization maximizes.
        Uses ONLY genetic distance data - no fake watermark scores.
        """
        try:
            if not languages:
                return 0.0

            # Get strategy evaluation function
            strategy_func = self.strategy_functions[self.strategy]

            # Evaluate using pure genetic distance
            result = strategy_func(languages, target_lang)

            # Track evaluation
            evaluation_record = {
                "iteration": len(self.evaluation_history) + 1,
                "languages": languages,
                "score": result.strategy_score,
                "strategy": self.strategy,
                "pivot_languages": result.pivot_languages,
                "genetic_distances": result.genetic_distances,
                "distance_features": result.distance_features
            }
            self.evaluation_history.append(evaluation_record)

            print(f"  Eval {len(self.evaluation_history)}: Languages {languages} -> Score: {result.strategy_score:.4f}")
            print(f"    Pivots: {result.pivot_languages[:4]}")
            print(f"    Mean distance: {result.distance_features['mean_distance']:.3f}")

            return result.strategy_score

        except Exception as e:
            print(f"Error evaluating languages {languages}: {e}")
            return 0.0

    def objective_function(self, x: List[str], max_languages: int = 10) -> float:
        """
        Objective function for scikit-optimize.
        Returns NEGATIVE score since skopt minimizes.
        """
        languages = self.decode_search_point(x)

        # Constraint: need at least 1 language
        if len(languages) == 0 or len(languages) > max_languages:
            return 1000.0  # Large penalty

        score = self.evaluate_language_combination(languages, self.current_target_lang)
        return -score  # Negative for minimization

    def optimize_languages(self, target_lang: str, n_calls: int = 25, max_languages: int = 10,
                          random_state: int = 42) -> Dict:
        """
        Run Bayesian optimization to find optimal language combinations.

        Args:
            target_lang: Target language (e.g., 'fra' for French attacks)
            n_calls: Total number of evaluations
            max_languages: Maximum languages per combination
            random_state: Random seed

        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*60}")
        print(f"GENETIC DISTANCE OPTIMIZATION: {self.strategy.upper()}")
        print(f"{'='*60}")
        print(f"Target language: {target_lang}")
        print(f"Strategy: {self.strategy}")
        print(f"Evaluations: {n_calls}")

        # Set context
        self.current_target_lang = target_lang
        self.evaluation_history = []

        # Create search space
        dimensions = self.create_search_space(max_languages)

        # Create objective function
        def objective(x):
            return self.objective_function(x, max_languages)

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
        best_languages = self.decode_search_point(best_x)
        best_score = -result.fun

        # Get detailed result for best combination
        best_evaluation = None
        for record in self.evaluation_history:
            if record["languages"] == best_languages:
                best_evaluation = record
                break

        optimization_result = {
            'strategy': self.strategy,
            'target_language': target_lang,
            'best_languages': best_languages,
            'best_score': best_score,
            'total_evaluations': len(self.evaluation_history),
            'best_evaluation': best_evaluation,
            'all_evaluations': self.evaluation_history,
            'skopt_result': result
        }

        print(f"\n=== OPTIMIZATION COMPLETE ({self.strategy.upper()}) ===")
        print(f"Best languages: {best_languages}")
        print(f"Best score: {best_score:.4f}")

        if best_evaluation:
            print(f"Best pivot languages: {best_evaluation['pivot_languages'][:5]}")
            features = best_evaluation['distance_features']
            print(f"Distance features:")
            print(f"  Mean distance: {features['mean_distance']:.3f}")
            print(f"  Max distance: {features['max_distance']:.3f}")
            print(f"  Distance diversity: {features['distance_diversity']:.3f}")

        return optimization_result


class GeneticDistanceComparator:
    """Compare all four genetic distance strategies"""

    def __init__(self, cluster_file: str = "clusters_genetic.csv"):
        self.cluster_file = cluster_file

    def compare_all_strategies(self, target_lang: str, n_calls: int = 20,
                             max_languages: int = 10) -> Dict:
        """
        Compare all four genetic distance optimization strategies.
        """
        print(f"\n{'='*80}")
        print(f"COMPARING ALL GENETIC DISTANCE STRATEGIES")
        print(f"{'='*80}")
        print(f"Target language: {target_lang}")
        print(f"Evaluations per strategy: {n_calls}")

        strategies = ['max_distance', 'min_distance', 'diversity', 'gp_learning']
        results = {}

        for strategy in strategies:
            print(f"\n>>> TESTING {strategy.upper().replace('_', ' ')} STRATEGY <<<")

            optimizer = GeneticDistanceOptimizer(self.cluster_file, strategy)
            strategy_result = optimizer.optimize_languages(target_lang, n_calls, max_languages)
            results[strategy] = strategy_result

        # Compare results
        print(f"\n{'='*60}")
        print(f"STRATEGY COMPARISON RESULTS")
        print(f"{'='*60}")

        best_strategy = None
        best_score = -np.inf

        for strategy, result in results.items():
            score = result['best_score']
            languages = result['best_languages']

            print(f"\n{strategy.upper().replace('_', ' ')} Strategy:")
            print(f"  Best languages: {languages}")
            print(f"  Score: {score:.4f}")

            if result['best_evaluation']:
                pivots = result['best_evaluation']['pivot_languages'][:3]
                print(f"  Key pivots: {pivots}")

            if score > best_score:
                best_score = score
                best_strategy = strategy

        print(f"\n🏆 BEST STRATEGY: {best_strategy.upper().replace('_', ' ')}")
        print(f"📈 Best score: {best_score:.4f}")

        results['comparison'] = {
            'best_strategy': best_strategy,
            'best_score': best_score,
            'target_language': target_lang
        }

        return results

    def save_results(self, results: Dict, filename: str):
        """Save comparison results to JSON"""

        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Convert to serializable format
        serializable_results = {}

        for strategy, result in results.items():
            if strategy == 'comparison':
                serializable_results[strategy] = convert_numpy_types(result)
            else:
                serializable_results[strategy] = {
                    'strategy': result['strategy'],
                    'target_language': result['target_language'],
                    'best_languages': convert_numpy_types(result['best_languages']),
                    'best_score': float(result['best_score']),
                    'total_evaluations': int(result['total_evaluations']),
                    'best_evaluation': convert_numpy_types(result['best_evaluation'])
                }

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # Test all strategies for French attacks
    comparator = GeneticDistanceComparator()

    results = comparator.compare_all_strategies(
        target_lang="fra",
        n_calls=15,
        max_clusters=2
    )

    comparator.save_results(results, "genetic_distance_comparison.json")

    print(f"\n🔬 Genetic Distance Optimization Complete!")
    print(f"All strategies tested using ONLY real genetic distance data from URIEL.")
    print(f"No assumptions made about watermark effectiveness.")