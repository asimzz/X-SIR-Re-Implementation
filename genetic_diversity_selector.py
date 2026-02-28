#!/usr/bin/env python3
"""
Genetic Diversity-Based Language Selector

Selects 3 languages from supported_languages.txt that are maximally diverse
in terms of their URIEL genetic feature vectors.

Author: Asim
"""

import numpy as np
from typing import List, Tuple
import random
from itertools import combinations
from uriel_genetic_distance import URIELGeneticDistance


class GeneticDiversitySelector:
    """
    Selects languages that are maximally diverse based on URIEL genetic features.
    """

    def __init__(self, languages_file: str = "supported_languages.txt", random_seed: int = 42):
        """
        Initialize the genetic diversity selector.

        Args:
            languages_file: Path to file containing supported language codes
            random_seed: Random seed for reproducible selection
        """
        self.languages_file = languages_file
        self.random_seed = random_seed
        self.uriel_distance = URIELGeneticDistance()

        # Load supported languages
        self.supported_languages = self._load_supported_languages()

        # Filter to languages available in URIEL
        self.available_languages = self._filter_uriel_languages()

        print(f"Loaded {len(self.supported_languages)} supported languages")
        print(f"Available in URIEL: {len(self.available_languages)} languages")

    def _load_supported_languages(self) -> List[str]:
        """Load language codes from supported_languages.txt"""
        with open(self.languages_file, 'r', encoding='utf-8') as f:
            languages = [line.strip() for line in f if line.strip()]
        return languages

    def _filter_uriel_languages(self) -> List[str]:
        """Filter languages to only those available in URIEL database"""
        available = []
        for lang in self.supported_languages:
            if self.uriel_distance.is_language_supported(lang):
                available.append(lang)
            else:
                print(f"Warning: {lang} not in URIEL database")
        return available

    def calculate_diversity_score(self, language_triple: Tuple[str, str, str]) -> float:
        """
        Calculate genetic diversity score for a triple of languages.

        Args:
            language_triple: Triple of language codes

        Returns:
            Diversity score (sum of pairwise genetic distances)
        """
        lang1, lang2, lang3 = language_triple

        try:
            # Calculate pairwise genetic distances
            dist_12 = self.uriel_distance.get_genetic_distance(lang1, lang2)
            dist_13 = self.uriel_distance.get_genetic_distance(lang1, lang3)
            dist_23 = self.uriel_distance.get_genetic_distance(lang2, lang3)

            # Diversity score = sum of pairwise distances
            diversity_score = dist_12 + dist_13 + dist_23

            return diversity_score

        except Exception as e:
            print(f"Error calculating diversity for {language_triple}: {e}")
            return 0.0

    def select_diverse_languages(self, n_languages: int = 3, method: str = "exhaustive") -> Tuple[List[str], float]:
        """
        Select n languages that maximize genetic diversity.

        Args:
            n_languages: Number of languages to select
            method: Selection method ("exhaustive" or "random_sample")

        Returns:
            Tuple of (selected_languages, diversity_score)
        """
        if len(self.available_languages) < n_languages:
            raise ValueError(f"Only {len(self.available_languages)} languages available, cannot select {n_languages}")

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if method == "exhaustive":
            return self._exhaustive_selection(n_languages)
        elif method == "random_sample":
            return self._random_sample_selection(n_languages)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _exhaustive_selection(self, n_languages: int) -> Tuple[List[str], float]:
        """Exhaustively search all combinations for maximum diversity"""
        print(f"Exhaustive search over {len(list(combinations(self.available_languages, n_languages)))} combinations...")

        best_languages = None
        best_diversity = -1.0

        for language_combo in combinations(self.available_languages, n_languages):
            diversity_score = self.calculate_diversity_score(language_combo)

            if diversity_score > best_diversity:
                best_diversity = diversity_score
                best_languages = list(language_combo)

        return best_languages, best_diversity

    def _random_sample_selection(self, n_languages: int, n_samples: int = 100) -> Tuple[List[str], float]:
        """Randomly sample combinations for maximum diversity (faster for large sets)"""
        print(f"Random sampling {n_samples} combinations...")

        best_languages = None
        best_diversity = -1.0

        for _ in range(n_samples):
            # Randomly sample n languages
            language_combo = tuple(random.sample(self.available_languages, n_languages))
            diversity_score = self.calculate_diversity_score(language_combo)

            if diversity_score > best_diversity:
                best_diversity = diversity_score
                best_languages = list(language_combo)

        return best_languages, best_diversity

    def analyze_selection(self, selected_languages: List[str]) -> dict:
        """
        Analyze the genetic diversity of selected languages.

        Args:
            selected_languages: List of selected language codes

        Returns:
            Dictionary with diversity analysis
        """
        if len(selected_languages) < 2:
            return {"error": "Need at least 2 languages for analysis"}

        analysis = {
            "languages": selected_languages,
            "pairwise_distances": {},
            "min_distance": float('inf'),
            "max_distance": 0.0,
            "mean_distance": 0.0,
            "total_diversity": 0.0
        }

        # Calculate all pairwise distances
        distances = []
        for i in range(len(selected_languages)):
            for j in range(i + 1, len(selected_languages)):
                lang1, lang2 = selected_languages[i], selected_languages[j]
                try:
                    distance = self.uriel_distance.get_genetic_distance(lang1, lang2)
                    analysis["pairwise_distances"][f"{lang1}-{lang2}"] = distance
                    distances.append(distance)
                except Exception as e:
                    print(f"Error calculating distance {lang1}-{lang2}: {e}")

        if distances:
            analysis["min_distance"] = min(distances)
            analysis["max_distance"] = max(distances)
            analysis["mean_distance"] = np.mean(distances)
            analysis["total_diversity"] = sum(distances)

        return analysis


def main():
    """Test the genetic diversity selector"""
    print("Genetic Diversity-Based Language Selector")
    print("=" * 50)

    # Initialize selector
    selector = GeneticDiversitySelector()

    if len(selector.available_languages) < 3:
        print(f"Error: Only {len(selector.available_languages)} languages available in URIEL")
        return

    # Method selection based on number of languages
    n_combinations = len(list(combinations(selector.available_languages, 3)))
    print(f"Total possible combinations: {n_combinations}")

    # Use exhaustive search if feasible, otherwise random sampling
    if n_combinations <= 50000:
        method = "exhaustive"
    else:
        method = "random_sample"

    print(f"Using {method} selection method")

    # Select 3 most diverse languages
    selected_languages, diversity_score = selector.select_diverse_languages(
        n_languages=3,
        method=method
    )

    # Analyze results
    print(f"\n🎯 Selected Languages: {selected_languages}")
    print(f"🔢 Total Diversity Score: {diversity_score:.4f}")

    # Detailed analysis
    analysis = selector.analyze_selection(selected_languages)
    print(f"\n📊 Diversity Analysis:")
    print(f"   Mean pairwise distance: {analysis['mean_distance']:.4f}")
    print(f"   Min pairwise distance:  {analysis['min_distance']:.4f}")
    print(f"   Max pairwise distance:  {analysis['max_distance']:.4f}")

    print(f"\n🔍 Pairwise Distances:")
    for pair, distance in analysis["pairwise_distances"].items():
        print(f"   {pair}: {distance:.4f}")

    # Save results
    output_file = "selected_diverse_languages.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Genetically diverse languages selected for dynamic pivot selection\n")
        f.write(f"# Total diversity score: {diversity_score:.4f}\n")
        f.write(f"# Selection method: {method}\n")
        f.write(f"# Mean pairwise distance: {analysis['mean_distance']:.4f}\n")
        f.write("\n")
        for lang in selected_languages:
            f.write(f"{lang}\n")

    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()