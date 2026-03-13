"""
URIEL language-space utilities built on top of lang2vec.

This module exposes per-language feature vectors so languages can be compared
directly in a multidimensional feature space.
"""

import lang2vec.lang2vec as l2v
import numpy as np


class URIELLanguageSpace:
    """Access URIEL language feature vectors."""
    
    def __init__(self):
        """Initialize with lang2vec (contains URIEL data)."""
        # Cache per-language feature vectors keyed by (lang, feature_sets)
        self._feature_cache = {}
        
        # Get all available languages
        self.available_languages = set(l2v.available_languages())
        
    def get_feature_vector(self, lang: str, feature_sets=("geo",)) -> np.ndarray:
        """
        Return the concatenated URIEL feature vector for a language.

        Args:
            lang: ISO 639-3 language code.
            feature_sets: Iterable of lang2vec feature set names. By default we
                use "geo", which provides a continuous feature vector suitable
                for BO in language feature space.
        """
        if lang not in self.available_languages:
            raise ValueError(f"Language {lang} not in URIEL database")

        normalized_feature_sets = tuple(feature_sets)
        cache_key = (lang, normalized_feature_sets)
        if cache_key not in self._feature_cache:
            vectors = []
            for feature_set in normalized_feature_sets:
                features = l2v.get_features([lang], feature_set)
                vectors.append(np.asarray(features[lang], dtype=float))
            self._feature_cache[cache_key] = np.concatenate(vectors)

        return self._feature_cache[cache_key]

    def is_language_supported(self, lang: str) -> bool:
        """Check if language has URIEL feature data."""
        return lang in self.available_languages


if __name__ == "__main__":
    print("Testing URIEL Language Space Utilities")
    print("=" * 50)
    
    language_space = URIELLanguageSpace()

    test_languages = ["eng", "fra", "deu"]
    print("\nExample feature vectors (geo):")
    for lang in test_languages:
        try:
            vector = language_space.get_feature_vector(lang, ("geo",))
            print(f"  {lang}: shape={vector.shape}, first3={vector[:3]}")
        except Exception as e:
            print(f"  {lang}: ERROR - {e}")
    
    print(f"\nTotal languages in URIEL database: {len(language_space.available_languages)}")
    print(f"Example languages: {list(language_space.available_languages)[:20]}")
