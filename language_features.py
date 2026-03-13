"""
Language Feature Vectors from URIEL via lang2vec.

Provides combined syntactic + phonological feature vectors for languages,
used as the search space for Bayesian Optimization in STEAM BO.

Feature sets:
  - syntax_knn (103-D): word order, morphology, case marking, etc.
  - phonology_knn (28-D): phonological system properties
  - Combined (131-D): concatenation of both
"""

import lang2vec.lang2vec as l2v
import numpy as np
from typing import List


class LanguageFeatures:
    """Retrieve and cache language feature vectors from URIEL."""

    def __init__(self, feature_sets: List[str] = None):
        """
        Initialize with lang2vec.

        Args:
            feature_sets: List of lang2vec feature sets to concatenate.
                          Defaults to ['syntax_knn', 'phonology_knn'].
        """
        if feature_sets is None:
            feature_sets = ['syntax_knn', 'phonology_knn']
        self.feature_sets = feature_sets
        self._feature_cache = {}
        self.available_languages = set(l2v.available_languages())

    def get_feature_vector(self, lang: str) -> np.ndarray:
        """Get cached concatenated feature vector for a language (ISO 639-3 code)."""
        if lang not in self._feature_cache:
            vectors = []
            for fs in self.feature_sets:
                features = l2v.get_features([lang], fs)
                vectors.append(np.array(features[lang]))
            self._feature_cache[lang] = np.concatenate(vectors)
        return self._feature_cache[lang]

    def get_distance(self, lang1: str, lang2: str) -> float:
        """
        Euclidean distance between two languages in feature space.

        Args:
            lang1: First language code (ISO 639-3)
            lang2: Second language code (ISO 639-3)

        Returns:
            Euclidean distance between feature vectors
        """
        if lang1 == lang2:
            return 0.0
        if lang1 not in self.available_languages:
            raise ValueError(f"Language {lang1} not in URIEL database")
        if lang2 not in self.available_languages:
            raise ValueError(f"Language {lang2} not in URIEL database")

        vec1 = self.get_feature_vector(lang1)
        vec2 = self.get_feature_vector(lang2)
        return float(np.linalg.norm(vec1 - vec2))

    def is_language_supported(self, lang: str) -> bool:
        """Check if language is available in URIEL."""
        return lang in self.available_languages


if __name__ == "__main__":
    print("Testing Language Features (syntax_knn + phonology_knn)")
    print("=" * 50)

    lf = LanguageFeatures()

    test_pairs = [
        ('eng', 'deu', 'English-German (both Germanic)'),
        ('eng', 'fra', 'English-French (Germanic-Romance)'),
        ('fra', 'spa', 'French-Spanish (both Romance)'),
        ('eng', 'zho', 'English-Chinese (very different)'),
        ('deu', 'nld', 'German-Dutch (both Germanic)'),
        ('rus', 'pol', 'Russian-Polish (both Slavic)'),
    ]

    print(f"\nFeature sets: {lf.feature_sets}")
    vec = lf.get_feature_vector('eng')
    print(f"Combined feature dimensions: {len(vec)}")

    print("\nPairwise distances:")
    for lang1, lang2, description in test_pairs:
        try:
            dist = lf.get_distance(lang1, lang2)
            print(f"  {lang1}-{lang2}: {dist:.4f}  ({description})")
        except Exception as e:
            print(f"  {lang1}-{lang2}: ERROR - {e}")

    print(f"\nTotal languages in URIEL: {len(lf.available_languages)}")
