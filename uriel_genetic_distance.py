"""
Genetic Distance Calculator using URIEL data from lang2vec.

Uses language family feature vectors from URIEL database to compute
genetic distances between languages - NO CLUSTERS involved.
"""

import lang2vec.lang2vec as l2v
import numpy as np


class URIELGeneticDistance:
    """Compute genetic distances between languages using URIEL family vectors."""
    
    def __init__(self):
        """Initialize with lang2vec (contains URIEL data)."""
        # Cache for feature vectors
        self._feature_cache = {}
        
        # Get all available languages
        self.available_languages = set(l2v.available_languages())
        
    def get_genetic_distance(self, lang1: str, lang2: str) -> float:
        """
        Get genetic distance between two languages using URIEL family vectors.
        
        Args:
            lang1: First language code (ISO 639-3)
            lang2: Second language code (ISO 639-3)
            
        Returns:
            float: Genetic distance (Euclidean distance between family vectors)
                   Range approximately [0, 5], where:
                   - 0 = same language
                   - ~2.5 = closely related (e.g., eng-deu)
                   - ~4.0 = different families (e.g., eng-fra)
                   - ~5.0 = very distant (e.g., eng-zho)
        """
        # Same language
        if lang1 == lang2:
            return 0.0
            
        # Check if languages are available
        if lang1 not in self.available_languages:
            raise ValueError(f"Language {lang1} not in URIEL database")
        if lang2 not in self.available_languages:
            raise ValueError(f"Language {lang2} not in URIEL database")
        
        # Get feature vectors (with caching)
        vec1 = self._get_family_vector(lang1)
        vec2 = self._get_family_vector(lang2)
        
        # Compute Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        
        return float(distance)
    
    def _get_family_vector(self, lang: str) -> np.ndarray:
        """Get cached family feature vector for a language."""
        if lang not in self._feature_cache:
            # Get family features from lang2vec
            features = l2v.get_features([lang], 'fam')
            self._feature_cache[lang] = np.array(features[lang])
        
        return self._feature_cache[lang]
    
    def is_language_supported(self, lang: str) -> bool:
        """Check if language has URIEL genetic data."""
        return lang in self.available_languages
    
    def get_distance_matrix(self, languages: list) -> np.ndarray:
        """
        Compute pairwise distance matrix for a list of languages.
        
        Args:
            languages: List of language codes
            
        Returns:
            numpy array of shape (n_languages, n_languages) with pairwise distances
        """
        n = len(languages)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.get_genetic_distance(languages[i], languages[j])
                matrix[i, j] = dist
                matrix[j, i] = dist
                
        return matrix


if __name__ == "__main__":
    # Test the genetic distance calculator
    print("Testing URIEL Genetic Distance Calculator")
    print("=" * 50)
    
    gd = URIELGeneticDistance()
    
    # Test some language pairs
    test_pairs = [
        ('eng', 'deu', 'English-German (both Germanic)'),
        ('eng', 'fra', 'English-French (Germanic-Romance)'),
        ('fra', 'spa', 'French-Spanish (both Romance)'),
        ('eng', 'zho', 'English-Chinese (very different)'),
        ('deu', 'nld', 'German-Dutch (both Germanic)'),
        ('rus', 'pol', 'Russian-Polish (both Slavic)'),
    ]
    
    print("\nGenetic distances:")
    for lang1, lang2, description in test_pairs:
        try:
            distance = gd.get_genetic_distance(lang1, lang2)
            print(f"  {lang1}-{lang2}: {distance:.4f}  ({description})")
        except Exception as e:
            print(f"  {lang1}-{lang2}: ERROR - {e}")
    
    print(f"\nTotal languages in URIEL database: {len(gd.available_languages)}")
    print(f"Example languages: {list(gd.available_languages)[:20]}")