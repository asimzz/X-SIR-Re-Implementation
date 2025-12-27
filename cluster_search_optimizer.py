#!/usr/bin/env python3
"""
Bayesian Optimization for Language Cluster Search using scikit-optimize

This module implements Bayesian optimization to find optimal language clusters
for improving watermark detection via back translation attacks.
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import itertools
import random

# Scikit-optimize for Bayesian optimization
from skopt import gp_minimize
from skopt.space import Categorical

from cluster_evaluator import STEAMEvaluator


@dataclass
class ClusterSearchResult:
    """Result of cluster search optimization"""
    best_clusters: List[int]
    best_score: float
    total_evaluations: int
    languages_in_best_clusters: List[List[str]]
    all_evaluations: List[Tuple[List[int], float]]
    optimization_result: object  # skopt result object


class LanguageClusterSearcher:
    """
    Bayesian optimization for finding optimal language cluster combinations.

    Uses scikit-optimize's Gaussian Process optimization to efficiently search
    through the combinatorial space of language cluster combinations.
    """

    def __init__(self, cluster_file: str = "clusters_genetic.csv", max_clusters_total: int = 20):
        """
        Initialize the cluster searcher.

        Args:
            cluster_file: Path to CSV file containing language cluster assignments
            max_clusters_total: Maximum total number of clusters to consider
        """
        self.cluster_file = cluster_file
        self.max_clusters_total = max_clusters_total
        self.cluster_info = None
        self.available_clusters = None
        self.cluster_to_languages = {}

        # For tracking during optimization
        self.current_watermark_dir = None
        self.current_target_lang = None
        self.evaluation_history = []

        # Load cluster information
        self._load_cluster_info()

    def _load_cluster_info(self):
        """Load cluster information from CSV file"""
        try:
            df = pd.read_csv(self.cluster_file)
            self.cluster_info = df

            # Get available clusters (limit to reasonable number)
            available_clusters = sorted(df['cluster_id'].unique())
            self.available_clusters = available_clusters[:self.max_clusters_total]

            # Create mapping from cluster_id to languages
            for cluster_id in self.available_clusters:
                languages = df[df['cluster_id'] == cluster_id]['language'].tolist()
                self.cluster_to_languages[cluster_id] = languages

            print(f"Loaded {len(self.available_clusters)} clusters with {len(df)} languages total")
            print(f"Available clusters: {self.available_clusters}")

        except FileNotFoundError:
            print(f"Warning: Cluster file {self.cluster_file} not found. Using simulated clusters.")
            self._create_simulated_clusters()

    def _create_simulated_clusters(self):
        """Create simulated cluster data for testing"""

        # Simulate language families as clusters
        simulated_clusters = {
            0: ['eng', 'deu', 'nld'],  # Germanic
            1: ['spa', 'fra', 'ita', 'por'],  # Romance
            2: ['rus', 'pol', 'ukr'],  # Slavic
            3: ['hin', 'ben', 'mar'],  # Indo-Aryan
            4: ['ara', 'heb'],  # Semitic
            5: ['jpn'],  # Japanese (isolate)
            6: ['kor'],  # Korean (isolate)
            7: ['zho'],  # Chinese
            8: ['vie'],  # Vietnamese
            9: ['tha'],  # Thai
            10: ['tur'],  # Turkish
            11: ['fin', 'hun', 'est'],  # Finno-Ugric
            12: ['swe', 'nor', 'dan'],  # North Germanic
            13: ['ell'],  # Greek
            14: ['fas'],  # Persian
        }

        self.cluster_to_languages = simulated_clusters
        self.available_clusters = list(simulated_clusters.keys())
        print(f"Created {len(self.available_clusters)} simulated language clusters")

    def evaluate_cluster_combination(self, clusters: List[int]) -> float:
        """
        Evaluate a cluster combination using STEAM method.

        This is the objective function that Bayesian optimization will try to maximize.
        """
        try:
            # Get languages in clusters
            cluster_to_languages = {}
            all_languages = []

            for cluster_id in clusters:
                if cluster_id in self.cluster_to_languages:
                    langs = self.cluster_to_languages[cluster_id]
                    cluster_to_languages[cluster_id] = langs
                    all_languages.extend(langs)

            if not cluster_to_languages:
                return 0.0

            # Use STEAM evaluator
            evaluator = STEAMEvaluator(self.current_watermark_dir)
            result = evaluator.evaluate_steam_method(
                cluster_combination=clusters,
                cluster_to_languages=cluster_to_languages,
                target_attack_lang=self.current_target_lang,
                normalization_method="z_score_max"
            )

            score = result.final_score

            # Track evaluation
            evaluation_record = {
                "iteration": len(self.evaluation_history) + 1,
                "clusters": [int(c) for c in clusters],
                "score": float(score),
                "languages": list(all_languages),
                "details": {
                    "final_score": float(result.final_score),
                    "baseline_auc": float(result.baseline_auc),
                    "pivot_languages": list(result.pivot_languages),
                    "back_translation_aucs": {k: float(v) for k, v in result.back_translation_aucs.items()}
                }
            }
            self.evaluation_history.append(evaluation_record)

            print(f"  Eval {len(self.evaluation_history)}: Clusters {clusters} -> Score: {score:.4f}")

            return score

        except Exception as e:
            print(f"Error evaluating clusters {clusters}: {e}")
            # Return a low score for failed evaluations
            return 0.1

    def _create_search_space(self, max_clusters_per_combo: int = 3, target_lang: str = None):
        """
        Create the search space for scikit-optimize.

        Uses genetic distance awareness to focus search on relevant clusters.
        """

        # Get target language's cluster for intelligent search space creation
        target_cluster = None
        if target_lang:
            try:
                df = pd.read_csv("clusters_genetic.csv")
                target_cluster_row = df[df['language'] == target_lang]
                if not target_cluster_row.empty:
                    target_cluster = target_cluster_row['cluster_id'].iloc[0]
                    print(f"Target language '{target_lang}' is in cluster {target_cluster}")
            except Exception as e:
                print(f"Could not load target cluster info: {e}")

        # Prioritize clusters based on genetic distance from target
        if target_cluster is not None:
            # Sort clusters by distance from target cluster (for genetic diversity)
            cluster_distances = []
            for cluster_id in self.available_clusters:
                distance = abs(cluster_id - target_cluster) if cluster_id != target_cluster else 999  # Avoid same cluster
                cluster_distances.append((cluster_id, distance))

            # Sort by distance (prioritize more distant clusters for better attack effectiveness)
            cluster_distances.sort(key=lambda x: x[1], reverse=True)

            # Take top clusters for search space (most genetically distant)
            limited_clusters = [cid for cid, _ in cluster_distances[:min(10, len(cluster_distances))]]
            print(f"Prioritized clusters for target {target_lang}: {limited_clusters[:5]}...")

        else:
            # Fallback: use first 10 clusters
            limited_clusters = self.available_clusters[:min(10, len(self.available_clusters))]

        # For each cluster position, allow selection from available clusters or "none"
        dimensions = []
        for i in range(max_clusters_per_combo):
            # Each position can select any cluster or -1 for "none"
            choices = [-1] + limited_clusters  # -1 means "no cluster selected for this position"
            dimensions.append(Categorical(choices, name=f'cluster_pos_{i}'))

        return dimensions

    def _decode_search_point(self, x: List[int]) -> List[int]:
        """Convert search space point to cluster list"""
        selected_clusters = []
        for cluster_id in x:
            if cluster_id != -1 and cluster_id not in selected_clusters:  # Avoid duplicates
                selected_clusters.append(cluster_id)
        return sorted(selected_clusters)

    def _objective_function(self, x: List[int], max_clusters: int = 3) -> float:
        """
        Objective function for scikit-optimize.

        Returns NEGATIVE score since skopt minimizes by default.
        """
        clusters = self._decode_search_point(x)

        # Constraint: limit number of clusters
        if len(clusters) == 0 or len(clusters) > max_clusters:
            # Track invalid evaluations too
            evaluation_record = {
                "iteration": len(self.evaluation_history) + 1,
                "clusters": [int(c) for c in clusters],
                "score": 0.0,
                "languages": [],
                "details": {"error": f"Invalid cluster count: {len(clusters)}"}
            }
            self.evaluation_history.append(evaluation_record)
            print(f"  Eval {len(self.evaluation_history)}: Clusters {clusters} -> INVALID (count: {len(clusters)})")
            return 1000.0  # Large penalty for invalid combinations

        # Evaluate and return negative score (for minimization)
        score = self.evaluate_cluster_combination(clusters)
        return -score  # Negative because skopt minimizes

    def bayesian_search(self, watermark_dir: str, target_lang: str,
                       n_calls: int = 25, max_clusters: int = 3,
                       acquisition_func: str = 'EI', random_state: int = 42) -> ClusterSearchResult:
        """
        Run Bayesian optimization to find optimal cluster combinations.

        Args:
            watermark_dir: Directory containing watermarked texts
            target_lang: Target language for attack
            n_calls: Total number of evaluations to perform
            max_clusters: Maximum clusters per combination
            acquisition_func: Acquisition function ('EI', 'PI', 'LCB')
            random_state: Random seed for reproducibility

        Returns:
            ClusterSearchResult with optimization results
        """
        print(f"Starting Bayesian optimization for cluster search")
        print(f"Target: {target_lang}, Evaluations: {n_calls}, Max clusters: {max_clusters}")
        print(f"Acquisition function: {acquisition_func}")

        # Set context for evaluation function
        self.current_watermark_dir = watermark_dir
        self.current_target_lang = target_lang
        self.evaluation_history = []

        # Create search space
        dimensions = self._create_search_space(max_clusters)

        # Map acquisition function names to skopt strings
        acq_func_map = {
            'EI': 'EI',
            'PI': 'PI',
            'LCB': 'LCB'
        }
        acq_func_str = acq_func_map.get(acquisition_func, 'EI')

        # Create objective function with max_clusters constraint
        def objective(x):
            return self._objective_function(x, max_clusters)

        # Run Bayesian optimization
        print(f"\n=== Running Bayesian Optimization ===")

        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=max(5, n_calls // 5),  # 20% for initial random sampling
            acq_func=acq_func_str,  # Use string name instead of function
            random_state=random_state,
            verbose=True,
            n_jobs=1  # Keep sequential for debugging
        )

        # Extract best result
        best_x = result.x
        best_clusters = self._decode_search_point(best_x)
        best_score = -result.fun  # Convert back to positive score

        # Get languages in best clusters
        languages_in_best = []
        for cluster_id in best_clusters:
            if cluster_id in self.cluster_to_languages:
                languages_in_best.append(self.cluster_to_languages[cluster_id])

        # Compile all evaluations
        all_evaluations = [(record["clusters"], record["score"])
                          for record in self.evaluation_history]

        search_result = ClusterSearchResult(
            best_clusters=best_clusters,
            best_score=best_score,
            total_evaluations=len(self.evaluation_history),
            languages_in_best_clusters=languages_in_best,
            all_evaluations=all_evaluations,
            optimization_result=result
        )

        print(f"\n=== Optimization Complete ===")
        print(f"Best clusters: {search_result.best_clusters}")
        print(f"Best score: {search_result.best_score:.4f}")
        print(f"Total evaluations: {search_result.total_evaluations}")
        print(f"Languages in best clusters:")
        for i, langs in enumerate(languages_in_best):
            cluster_id = best_clusters[i]
            print(f"  Cluster {cluster_id}: {langs}")

        return search_result

    def random_search_baseline(self, watermark_dir: str, target_lang: str,
                              n_calls: int = 25, max_clusters: int = 3,
                              random_state: int = 42) -> ClusterSearchResult:
        """
        Random search baseline for comparison with Bayesian optimization.
        """
        print(f"Running random search baseline...")
        print(f"Target: {target_lang}, Evaluations: {n_calls}, Max clusters: {max_clusters}")

        # Set context
        self.current_watermark_dir = watermark_dir
        self.current_target_lang = target_lang
        self.evaluation_history = []

        random.seed(random_state)

        best_score = -np.inf
        best_clusters = None

        for i in range(n_calls):
            # Generate random cluster combination
            n_clusters = random.randint(1, min(max_clusters, len(self.available_clusters)))
            clusters = random.sample(self.available_clusters, n_clusters)
            clusters = sorted(clusters)

            print(f"\nRandom eval {i+1}/{n_calls}: {clusters}")

            # Evaluate
            score = self.evaluate_cluster_combination(clusters)

            if score > best_score:
                best_score = score
                best_clusters = clusters
                print(f"  🎉 New best: {score:.4f}")

        # Get languages in best clusters
        languages_in_best = []
        if best_clusters:
            for cluster_id in best_clusters:
                if cluster_id in self.cluster_to_languages:
                    languages_in_best.append(self.cluster_to_languages[cluster_id])

        all_evaluations = [(record["clusters"], record["score"])
                          for record in self.evaluation_history]

        return ClusterSearchResult(
            best_clusters=best_clusters or [],
            best_score=best_score,
            total_evaluations=len(self.evaluation_history),
            languages_in_best_clusters=languages_in_best,
            all_evaluations=all_evaluations,
            optimization_result=None
        )

    def save_results(self, results: ClusterSearchResult, filename: str):
        """Save optimization results to JSON file"""

        # Convert skopt result to serializable format
        opt_result_dict = None
        if results.optimization_result:
            opt_result_dict = {
                'x': results.optimization_result.x,
                'fun': results.optimization_result.fun,
                'func_vals': results.optimization_result.func_vals.tolist(),
                'x_iters': [x for x in results.optimization_result.x_iters],
                'n_calls': len(results.optimization_result.x_iters)
            }

        results_dict = {
            "best_clusters": [int(c) for c in results.best_clusters],
            "best_score": float(results.best_score),
            "total_evaluations": int(results.total_evaluations),
            "languages_in_best_clusters": results.languages_in_best_clusters,
            "all_evaluations": [[list(clusters), float(score)] for clusters, score in results.all_evaluations],
            "evaluation_history": self.evaluation_history,
            "optimization_result": opt_result_dict,
            "cluster_to_languages": {str(k): v for k, v in self.cluster_to_languages.items()},
            "available_clusters": [int(c) for c in self.available_clusters]
        }

        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {filename}")

    def compare_methods(self, watermark_dir: str, target_lang: str,
                       n_calls: int = 25, max_clusters: int = 3) -> Dict:
        """
        Compare Bayesian optimization vs random search.
        """
        print(f"=== COMPARING OPTIMIZATION METHODS ===")

        # Run Bayesian optimization
        print(f"\n1. Bayesian Optimization:")
        bayesian_results = self.bayesian_search(
            watermark_dir, target_lang, n_calls, max_clusters
        )

        # Run random search
        print(f"\n2. Random Search Baseline:")
        random_results = self.random_search_baseline(
            watermark_dir, target_lang, n_calls, max_clusters
        )

        # Compare results
        print(f"\n=== COMPARISON RESULTS ===")
        print(f"Bayesian Optimization:")
        print(f"  Best score: {bayesian_results.best_score:.4f}")
        print(f"  Best clusters: {bayesian_results.best_clusters}")

        print(f"Random Search:")
        print(f"  Best score: {random_results.best_score:.4f}")
        print(f"  Best clusters: {random_results.best_clusters}")

        improvement = bayesian_results.best_score - random_results.best_score
        print(f"Improvement: {improvement:.4f} ({improvement/random_results.best_score*100:.1f}%)")

        return {
            "bayesian": bayesian_results,
            "random": random_results,
            "improvement": improvement
        }


if __name__ == "__main__":
    # Example usage
    searcher = LanguageClusterSearcher()

    # Run comparison
    comparison = searcher.compare_methods(
        watermark_dir="gen/llama-3.2-1B/kgw_seed0",
        target_lang="fr",
        n_calls=20,
        max_clusters=2
    )

    # Save results
    searcher.save_results(comparison["bayesian"], "bayesian_optimization_results.json")
    searcher.save_results(comparison["random"], "random_search_results.json")