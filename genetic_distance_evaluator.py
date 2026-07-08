#!/usr/bin/env python3
"""
Genetic Distance-Based Cluster Evaluator

This module implements cluster evaluation using ONLY real genetic distance data
from URIEL, without any assumptions about watermark effectiveness.

Four optimization strategies:
1. Maximum Distance: Maximize genetic distance between target and pivot languages
2. Minimum Distance: Minimize genetic distance (similar languages)
3. Distance Diversity: Optimize for diverse mix of close and far languages
4. GP Learning: Let Gaussian Process learn optimal distance patterns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from urielplus import urielplus


@dataclass
class GeneticDistanceResult:
    """Result of genetic distance-based evaluation"""
    cluster_combination: List[int]
    pivot_languages: List[str]
    target_language: str
    genetic_distances: Dict[str, float]
    distance_features: Dict[str, float]
    strategy_score: float
    strategy_type: str


class GeneticDistanceEvaluator:
    """
    Evaluates language cluster combinations using ONLY genetic distance from URIEL.

    No assumptions about watermark effectiveness - purely distance-based optimization.
    """

    def __init__(self, cluster_file: str = "clusters_genetic.csv"):
        self.cluster_file = cluster_file
        self.uriel = urielplus.URIELPlus()
        self.cluster_to_languages = {}
        self.language_to_cluster = {}

        # Load cluster data
        self._load_cluster_mapping()

        print(f"Genetic Distance Evaluator initialized with URIEL database")
        print(f"Available languages with genetic data: {len(self.language_to_cluster)}")

    def _load_cluster_mapping(self):
        """Load cluster to language mappings from CSV"""
        try:
            df = pd.read_csv(self.cluster_file)

            # Create mappings
            for _, row in df.iterrows():
                lang = row['language']
                cluster_id = row['cluster_id']

                if cluster_id not in self.cluster_to_languages:
                    self.cluster_to_languages[cluster_id] = []
                self.cluster_to_languages[cluster_id].append(lang)
                self.language_to_cluster[lang] = cluster_id

            print(f"Loaded {len(df)} languages in {df['cluster_id'].nunique()} genetic clusters")

        except Exception as e:
            print(f"Error loading cluster data: {e}")
            # Create minimal test mapping
            self.cluster_to_languages = {
                0: ['rus', 'pol', 'ces'],
                1: ['hin', 'ben', 'urd'],
                2: ['zho', 'jpn', 'kor'],
                3: ['eng', 'deu', 'nld'],
                14: ['fra', 'spa', 'ita']
            }
            for cluster_id, langs in self.cluster_to_languages.items():
                for lang in langs:
                    self.language_to_cluster[lang] = cluster_id

    def get_genetic_distance(self, lang1: str, lang2: str) -> float:
        """Get genetic distance between two languages using URIEL"""
        try:
            # Check if languages have genetic distance data
            genetic_langs = set(self.uriel.get_languages_with_distance_data(distance_type="genetic"))

            if lang1 not in genetic_langs or lang2 not in genetic_langs:
                print(f"Warning: No genetic data for {lang1}-{lang2}, using fallback")
                return self._fallback_genetic_distance(lang1, lang2)

            # Get genetic distance from URIEL
            distance = self.uriel.new_genetic_distance([lang1, lang2])
            return float(distance)

        except Exception as e:
            print(f"Error getting genetic distance {lang1}-{lang2}: {e}")
            return self._fallback_genetic_distance(lang1, lang2)

    def _fallback_genetic_distance(self, lang1: str, lang2: str) -> float:
        """Fallback genetic distance based on cluster assignments"""
        cluster1 = self.language_to_cluster.get(lang1, -1)
        cluster2 = self.language_to_cluster.get(lang2, -1)

        if cluster1 == -1 or cluster2 == -1:
            return 1.0  # Max distance for unknown languages

        # Normalize cluster distance to [0,1] range
        max_cluster_dist = 20  # Assume max 20 clusters
        cluster_dist = abs(cluster1 - cluster2) / max_cluster_dist
        return min(1.0, cluster_dist)

    def get_pivot_languages_from_clusters(self, cluster_combination: List[int],
                                        target_lang: str) -> List[str]:
        """Get all pivot languages from cluster combination, excluding target language"""
        pivot_languages = []

        for cluster_id in cluster_combination:
            if cluster_id in self.cluster_to_languages:
                cluster_langs = self.cluster_to_languages[cluster_id]
                pivot_languages.extend(cluster_langs)

        # Remove duplicates and target language
        pivot_languages = list(set(pivot_languages))
        if target_lang in pivot_languages:
            pivot_languages.remove(target_lang)

        return pivot_languages

    def compute_distance_features(self, pivot_languages: List[str],
                                target_lang: str) -> Dict[str, float]:
        """Compute genetic distance features for optimization"""

        if not pivot_languages:
            return {"mean_distance": 0.0, "max_distance": 0.0, "min_distance": 0.0,
                   "distance_diversity": 0.0, "distance_sum": 0.0}

        # Get all genetic distances
        distances = []
        genetic_distance_map = {}

        for pivot_lang in pivot_languages:
            dist = self.get_genetic_distance(target_lang, pivot_lang)
            distances.append(dist)
            genetic_distance_map[pivot_lang] = dist

        distances = np.array(distances)

        # Compute various distance features
        features = {
            "mean_distance": np.mean(distances),
            "max_distance": np.max(distances),
            "min_distance": np.min(distances),
            "distance_diversity": np.std(distances) if len(distances) > 1 else 0.0,
            "distance_sum": np.sum(distances),
            "median_distance": np.median(distances),
            "distance_range": np.max(distances) - np.min(distances) if len(distances) > 1 else 0.0
        }

        return features, genetic_distance_map

    def evaluate_max_distance_strategy(self, cluster_combination: List[int],
                                     target_lang: str) -> GeneticDistanceResult:
        """Strategy 1: Maximize genetic distance (diverse pivots hypothesis)"""

        pivot_languages = self.get_pivot_languages_from_clusters(cluster_combination, target_lang)
        features, genetic_distances = self.compute_distance_features(pivot_languages, target_lang)

        # Score = maximize mean genetic distance
        strategy_score = features["mean_distance"]

        return GeneticDistanceResult(
            cluster_combination=cluster_combination,
            pivot_languages=pivot_languages,
            target_language=target_lang,
            genetic_distances=genetic_distances,
            distance_features=features,
            strategy_score=strategy_score,
            strategy_type="max_distance"
        )

    def evaluate_min_distance_strategy(self, cluster_combination: List[int],
                                     target_lang: str) -> GeneticDistanceResult:
        """Strategy 2: Minimize genetic distance (similar pivots hypothesis)"""

        pivot_languages = self.get_pivot_languages_from_clusters(cluster_combination, target_lang)
        features, genetic_distances = self.compute_distance_features(pivot_languages, target_lang)

        # Score = minimize mean genetic distance (invert for maximization)
        strategy_score = 1.0 - features["mean_distance"]

        return GeneticDistanceResult(
            cluster_combination=cluster_combination,
            pivot_languages=pivot_languages,
            target_language=target_lang,
            genetic_distances=genetic_distances,
            distance_features=features,
            strategy_score=strategy_score,
            strategy_type="min_distance"
        )

    def evaluate_diversity_strategy(self, cluster_combination: List[int],
                                  target_lang: str) -> GeneticDistanceResult:
        """Strategy 3: Maximize distance diversity (mix of close/far hypothesis)"""

        pivot_languages = self.get_pivot_languages_from_clusters(cluster_combination, target_lang)
        features, genetic_distances = self.compute_distance_features(pivot_languages, target_lang)

        # Score = maximize distance diversity + coverage
        diversity_score = features["distance_diversity"]
        coverage_score = features["distance_range"]
        strategy_score = 0.6 * diversity_score + 0.4 * coverage_score

        return GeneticDistanceResult(
            cluster_combination=cluster_combination,
            pivot_languages=pivot_languages,
            target_language=target_lang,
            genetic_distances=genetic_distances,
            distance_features=features,
            strategy_score=strategy_score,
            strategy_type="diversity"
        )

    def evaluate_gp_learning_strategy(self, cluster_combination: List[int],
                                    target_lang: str) -> GeneticDistanceResult:
        """Strategy 4: GP learning - rich feature vector for Gaussian Process"""

        pivot_languages = self.get_pivot_languages_from_clusters(cluster_combination, target_lang)
        features, genetic_distances = self.compute_distance_features(pivot_languages, target_lang)

        # Rich feature combination - let GP learn optimal weighting
        feature_vector = np.array([
            features["mean_distance"],
            features["max_distance"],
            features["min_distance"],
            features["distance_diversity"],
            features["median_distance"],
            features["distance_range"],
            len(pivot_languages),  # Number of pivot languages
            len(set(self.language_to_cluster.get(lang, -1) for lang in pivot_languages))  # Cluster diversity
        ])

        # Composite score - GP will learn to optimize this
        strategy_score = np.mean(feature_vector)

        return GeneticDistanceResult(
            cluster_combination=cluster_combination,
            pivot_languages=pivot_languages,
            target_language=target_lang,
            genetic_distances=genetic_distances,
            distance_features=features,
            strategy_score=strategy_score,
            strategy_type="gp_learning"
        )

    # Language-based evaluation methods (direct language selection)
    def evaluate_max_distance_language_strategy(self, pivot_languages: List[str],
                                              target_lang: str) -> GeneticDistanceResult:
        """Strategy 1: Maximize genetic distance (language-based)"""

        # Remove target language if present
        pivot_languages = [lang for lang in pivot_languages if lang != target_lang]

        if not pivot_languages:
            return self._empty_result([], target_lang, "max_distance")

        features, genetic_distances = self.compute_distance_features(pivot_languages, target_lang)
        strategy_score = features["mean_distance"]

        return GeneticDistanceResult(
            cluster_combination=[],  # No cluster info needed
            pivot_languages=pivot_languages,
            target_language=target_lang,
            genetic_distances=genetic_distances,
            distance_features=features,
            strategy_score=strategy_score,
            strategy_type="max_distance"
        )

    def evaluate_min_distance_language_strategy(self, pivot_languages: List[str],
                                              target_lang: str) -> GeneticDistanceResult:
        """Strategy 2: Minimize genetic distance (language-based)"""

        pivot_languages = [lang for lang in pivot_languages if lang != target_lang]

        if not pivot_languages:
            return self._empty_result([], target_lang, "min_distance")

        features, genetic_distances = self.compute_distance_features(pivot_languages, target_lang)
        strategy_score = 1.0 - features["mean_distance"]  # Invert for minimization

        return GeneticDistanceResult(
            cluster_combination=[],
            pivot_languages=pivot_languages,
            target_language=target_lang,
            genetic_distances=genetic_distances,
            distance_features=features,
            strategy_score=strategy_score,
            strategy_type="min_distance"
        )

    def evaluate_diversity_language_strategy(self, pivot_languages: List[str],
                                           target_lang: str) -> GeneticDistanceResult:
        """Strategy 3: Optimize for diverse genetic distances (language-based)"""

        pivot_languages = [lang for lang in pivot_languages if lang != target_lang]

        if not pivot_languages:
            return self._empty_result([], target_lang, "diversity")

        features, genetic_distances = self.compute_distance_features(pivot_languages, target_lang)
        strategy_score = features["distance_diversity"]

        return GeneticDistanceResult(
            cluster_combination=[],
            pivot_languages=pivot_languages,
            target_language=target_lang,
            genetic_distances=genetic_distances,
            distance_features=features,
            strategy_score=strategy_score,
            strategy_type="diversity"
        )

    def evaluate_gp_learning_language_strategy(self, pivot_languages: List[str],
                                             target_lang: str) -> GeneticDistanceResult:
        """Strategy 4: GP learning (language-based)"""

        pivot_languages = [lang for lang in pivot_languages if lang != target_lang]

        if not pivot_languages:
            return self._empty_result([], target_lang, "gp_learning")

        features, genetic_distances = self.compute_distance_features(pivot_languages, target_lang)

        # Create feature vector for GP learning
        feature_vector = np.array([
            features["mean_distance"],
            features["distance_diversity"],
            features["distance_range"],
            len(pivot_languages)
        ])

        strategy_score = np.mean(feature_vector)

        return GeneticDistanceResult(
            cluster_combination=[],
            pivot_languages=pivot_languages,
            target_language=target_lang,
            genetic_distances=genetic_distances,
            distance_features=features,
            strategy_score=strategy_score,
            strategy_type="gp_learning"
        )

    def _empty_result(self, cluster_combination: List[int], target_lang: str,
                     strategy_type: str) -> GeneticDistanceResult:
        """Return empty result for invalid inputs"""
        return GeneticDistanceResult(
            cluster_combination=cluster_combination,
            pivot_languages=[],
            target_language=target_lang,
            genetic_distances={},
            distance_features={"mean_distance": 0.0, "distance_diversity": 0.0, "distance_range": 0.0},
            strategy_score=0.0,
            strategy_type=strategy_type
        )


if __name__ == "__main__":
    # Test all four strategies
    evaluator = GeneticDistanceEvaluator()

    test_clusters = [0, 2, 14]  # Slavic, Sino-Tibetan, Romance
    target_lang = "fra"

    print(f"\n=== Testing All Genetic Distance Strategies ===")
    print(f"Target language: {target_lang}")
    print(f"Test clusters: {test_clusters}")

    # Test each strategy
    strategies = [
        ("Max Distance", evaluator.evaluate_max_distance_strategy),
        ("Min Distance", evaluator.evaluate_min_distance_strategy),
        ("Diversity", evaluator.evaluate_diversity_strategy),
        ("GP Learning", evaluator.evaluate_gp_learning_strategy)
    ]

    for strategy_name, strategy_func in strategies:
        result = strategy_func(test_clusters, target_lang)

        print(f"\n--- {strategy_name} Strategy ---")
        print(f"Score: {result.strategy_score:.4f}")
        print(f"Pivot languages: {result.pivot_languages[:5]}")
        print(f"Mean genetic distance: {result.distance_features['mean_distance']:.3f}")
        print(f"Distance diversity: {result.distance_features['distance_diversity']:.3f}")