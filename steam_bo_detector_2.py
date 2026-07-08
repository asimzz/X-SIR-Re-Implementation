#!/usr/bin/env python3
"""
STEAM BO Per-Text Detector: Bayesian Optimization for Each Text

This module implements the supervisor's approach:
- For EACH text, run BO to find the best intermediate language
- Initial random sampling of languages → get z-scores  
- BO iteratively suggests next language based on genetic distance
- Use continuous genetic distance features from URIEL
- Search space: 108 languages (no clusters)

Author: Asim
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from skopt import gp_minimize
from skopt.space import Categorical

from realtime_backtranslation import RealtimeBacktranslator
from uriel_genetic_distance import URIELGeneticDistance


@dataclass
class PerTextDetectionResult:
    """Result of per-text BO-enhanced watermark detection."""
    text: str
    target_lang: str
    z_score: float
    best_intermediate_lang: str
    total_evaluations: int
    evaluation_history: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None


class SteamBOPerTextDetector:
    """
    Per-Text Bayesian Optimization Enhanced STEAM Watermark Detector.
    
    For each text:
    1. Random initial sampling of languages
    2. BO loop: suggest language → backtranslate → get z-score
    3. Use genetic distance to guide language selection
    4. Return best language found
    """
    
    def __init__(self,
                 watermark_detector,
                 languages_file: str = 'supported_languages.txt',
                 n_initial: int = 3,
                 max_evaluations: int = 8,
                 acquisition_func: str = 'EI',
                 random_state: int = 42):
        """
        Initialize per-text BO-enhanced STEAM detector.
        
        Args:
            watermark_detector: Watermark detector instance (XSIR/KGW/UW)
            languages_file: File containing supported language codes (default: 75 languages)
            n_initial: Number of random initial language evaluations
            max_evaluations: Maximum total language evaluations per text
            acquisition_func: BO acquisition function ('EI', 'UCB', 'PI')
            random_state: Random seed for reproducibility
        """
        self.watermark_detector = watermark_detector
        self.n_initial = n_initial
        self.max_evaluations = max_evaluations
        self.acquisition_func = acquisition_func
        self.random_state = random_state
        
        # Load all supported languages (deep_translator compatible)
        self.all_languages = self._load_languages(languages_file)
        print(f"Loaded {len(self.all_languages)} deep_translator-supported languages for BO optimization")
        
        # Initialize components
        self.backtranslator = RealtimeBacktranslator()
        self.genetic_distance_calc = URIELGeneticDistance()
        
        # Cache genetic distances
        self.genetic_distance_cache = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_languages(self, languages_file: str) -> List[str]:
        """Load supported language codes from file (deep_translator compatible)."""
        try:
            with open(languages_file, 'r') as f:
                languages = [line.strip() for line in f if line.strip()]
            return sorted(languages)
        except Exception as e:
            print(f"Error loading languages from {languages_file}: {e}")
            # Fallback: use a subset of common deep_translator supported languages
            return ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'jpn', 'kor', 'hin']
    
    def get_genetic_distance(self, lang1: str, lang2: str) -> float:
        """
        Get genetic distance between two languages using simple genetic distance calculator.
        
        Args:
            lang1: First language code
            lang2: Second language code
            
        Returns:
            Genetic distance in [0, 1] range (0=identical, 1=very distant)
        """
        # Check cache
        cache_key = tuple(sorted([lang1, lang2]))
        if cache_key in self.genetic_distance_cache:
            return self.genetic_distance_cache[cache_key]
        
        # Get genetic distance
        distance = self.genetic_distance_calc.get_genetic_distance(lang1, lang2)
        
        # Cache and return
        self.genetic_distance_cache[cache_key] = distance
        return distance
    
    def get_language_features(self, target_lang: str, intermediate_lang: str) -> np.ndarray:
        """
        Extract genetic distance features for a language pair.
        
        Args:
            target_lang: Target language code
            intermediate_lang: Intermediate language code
            
        Returns:
            Feature vector: [genetic_distance]
        """
        genetic_distance = self.get_genetic_distance(target_lang, intermediate_lang)
        return np.array([genetic_distance])
    
    def initial_random_sampling(self, text: str, target_lang: str) -> List[Dict[str, Any]]:
        """
        Perform initial random sampling of intermediate languages.
        
        Args:
            text: Input text for detection
            target_lang: Target language code
            
        Returns:
            List of evaluation results with (language, z_score) pairs
        """
        # Filter out target language
        available_langs = [lang for lang in self.all_languages if lang != target_lang]
        
        # Random sampling with text-specific seed for reproducibility
        rng = np.random.RandomState(self.random_state + hash(text[:50]) % 10000)
        n_to_sample = min(self.n_initial, len(available_langs))
        sampled_languages = rng.choice(available_langs, size=n_to_sample, replace=False)
        
        print(f"\n=== Initial Random Sampling ({n_to_sample} languages) ===")
        
        # Evaluate each sampled language
        evaluation_history = []
        for i, lang in enumerate(sampled_languages, 1):
            print(f"  [{i}/{n_to_sample}] Evaluating {lang}...", end=' ')
            
            result = self.backtranslator.translate_and_detect(
                text, target_lang, lang, self.watermark_detector
            )
            
            z_score = result['z_score'] if result['success'] else -np.inf
            genetic_dist = self.get_genetic_distance(target_lang, lang)
            
            evaluation_record = {
                'iteration': len(evaluation_history) + 1,
                'intermediate_lang': lang,
                'z_score': z_score,
                'genetic_distance': genetic_dist,
                'success': result['success'],
                'translation_result': result
            }
            
            evaluation_history.append(evaluation_record)
            print(f"z-score: {z_score:.4f}, genetic_dist: {genetic_dist:.3f}")
        
        return evaluation_history
    
    def create_bo_objective(self, text: str, target_lang: str,
                           evaluation_history: List[Dict[str, Any]]) -> Tuple:
        """
        Create BO objective function for this specific text.
        
        The objective function:
        1. Takes a language as input
        2. Checks if already evaluated (use cached result)
        3. If not, backtranslates and gets z-score
        4. Returns negative z-score (BO minimizes, we want to maximize z-score)
        
        Args:
            text: Text being evaluated (specific to this detection call)
            target_lang: Target language
            evaluation_history: Previous evaluations for THIS text
            
        Returns:
            Tuple of (objective_function, dimensions)
        """
        # Define search space: categorical over all available languages
        available_langs = [lang for lang in self.all_languages if lang != target_lang]
        dimensions = [Categorical(available_langs, name='intermediate_language')]
        
        def objective_function(x):
            """
            Objective function for BO.
            
            Args:
                x: List containing [intermediate_language]
                
            Returns:
                Negative z-score (for minimization)
            """
            intermediate_lang = x[0]  # Extract language from list
            
            # Check if this language was already evaluated for THIS text
            for record in evaluation_history:
                if record['intermediate_lang'] == intermediate_lang:
                    if record['success']:
                        return -record['z_score']  # Return cached result
                    else:
                        return 1000.0  # Large penalty for failed evaluations
            
            # Evaluate new language for THIS specific text
            print(f"  BO iteration {len(evaluation_history) + 1}: Trying {intermediate_lang}...", end=' ')
            
            result = self.backtranslator.translate_and_detect(
                text, target_lang, intermediate_lang, self.watermark_detector
            )
            
            z_score = result['z_score'] if result['success'] else -np.inf
            genetic_dist = self.get_genetic_distance(target_lang, intermediate_lang)
            
            # Store evaluation result for THIS text
            evaluation_record = {
                'iteration': len(evaluation_history) + 1,
                'intermediate_lang': intermediate_lang,
                'z_score': z_score,
                'genetic_distance': genetic_dist,
                'success': result['success'],
                'translation_result': result
            }
            
            evaluation_history.append(evaluation_record)
            print(f"z-score: {z_score:.4f}, genetic_dist: {genetic_dist:.3f}")
            
            if result['success']:
                return -z_score  # Negative for minimization (BO maximizes z-score)
            else:
                return 1000.0  # Large penalty for failed evaluations
        
        return objective_function, dimensions
    
    def detect_with_bo(self, text: str, target_lang: str) -> PerTextDetectionResult:
        """
        Perform per-text BO-enhanced watermark detection.
        
        For THIS specific text:
        1. Random initial sampling of languages → get z-scores
        2. BO loop: suggest language → backtranslate → get z-score
        3. Return best language found
        
        Args:
            text: Input text to check for watermarks
            target_lang: Target language code
            
        Returns:
            PerTextDetectionResult with best z-score and intermediate language
        """
        print(f"\n{'='*70}")
        print(f"STEAM BO Per-Text Detection")
        print(f"{'='*70}")
        print(f"Text: {text[:100]}...")
        print(f"Target language: {target_lang}")
        print(f"Max evaluations: {self.max_evaluations}")
        
        try:
            # Step 1: Initial random sampling for THIS text
            evaluation_history = self.initial_random_sampling(text, target_lang)
            
            if not evaluation_history:
                return PerTextDetectionResult(
                    text=text,
                    target_lang=target_lang,
                    z_score=-np.inf,
                    best_intermediate_lang=None,
                    total_evaluations=0,
                    evaluation_history=[],
                    success=False,
                    error="No initial evaluations completed"
                )
            
            # Step 2: BO optimization loop for THIS text
            n_bo_calls = max(0, self.max_evaluations - len(evaluation_history))
            
            if n_bo_calls > 0:
                print(f"\n=== Bayesian Optimization ({n_bo_calls} additional evaluations) ===")
                
                # Create BO objective for THIS text
                objective_func, dimensions = self.create_bo_objective(text, target_lang, evaluation_history)
                
                try:
                    # Prepare initial observations for BO
                    # Extract language names and objective values from evaluation history
                    initial_languages = [record['intermediate_lang'] for record in evaluation_history if record['success']]
                    initial_objectives = [-record['z_score'] for record in evaluation_history if record['success']]  # Negative for minimization

                    if len(initial_languages) == 0:
                        self.logger.warning("No successful initial evaluations for BO, skipping optimization")
                    else:
                        # Format as x0 and y0 for scikit-optimize
                        x0 = [[lang] for lang in initial_languages]  # List of lists (scikit-optimize format)
                        y0 = initial_objectives

                        self.logger.info(f"Providing {len(x0)} initial observations to BO")

                        # Run Bayesian optimization
                        result = gp_minimize(
                            func=objective_func,
                            dimensions=dimensions,
                            n_calls=n_bo_calls,
                            x0=x0,  # Provide initial observations
                            y0=y0,  # Provide initial objective values
                            acq_func=self.acquisition_func,
                            random_state=self.random_state,
                            verbose=False
                        )
                    
                except Exception as e:
                    self.logger.error(f"BO optimization failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Step 3: Find best result for THIS text
            best_result = None
            best_z_score = -np.inf
            
            for record in evaluation_history:
                if record['success'] and record['z_score'] > best_z_score:
                    best_z_score = record['z_score']
                    best_result = record
            
            if best_result is None:
                return PerTextDetectionResult(
                    text=text,
                    target_lang=target_lang,
                    z_score=-np.inf,
                    best_intermediate_lang=None,
                    total_evaluations=len(evaluation_history),
                    evaluation_history=evaluation_history,
                    success=False,
                    error="No successful evaluations"
                )
            
            print(f"\n{'='*70}")
            print(f"BEST RESULT FOR THIS TEXT")
            print(f"{'='*70}")
            print(f"Best intermediate language: {best_result['intermediate_lang']}")
            print(f"Best z-score: {best_z_score:.4f}")
            print(f"Genetic distance: {best_result['genetic_distance']:.3f}")
            print(f"Total evaluations: {len(evaluation_history)}")
            
            return PerTextDetectionResult(
                text=text,
                target_lang=target_lang,
                z_score=best_z_score,
                best_intermediate_lang=best_result['intermediate_lang'],
                total_evaluations=len(evaluation_history),
                evaluation_history=evaluation_history,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            
            return PerTextDetectionResult(
                text=text,
                target_lang=target_lang,
                z_score=-np.inf,
                best_intermediate_lang=None,
                total_evaluations=0,
                evaluation_history=[],
                success=False,
                error=str(e)
            )


def test_pertext_detector():
    """Test function for per-text STEAM BO detector."""
    # Mock watermark detector for testing
    class MockWatermarkDetector:
        def detect(self, text):
            # Return mock z-score with some randomness
            base_score = len(text) / 100.0
            noise = np.random.normal(0, 0.5)
            z_score = base_score + noise
            return {"z_score": z_score, "biases": []}
    
    # Initialize detector
    mock_detector = MockWatermarkDetector()
    steam_bo = SteamBOPerTextDetector(
        watermark_detector=mock_detector,
        n_initial=3,
        max_evaluations=6
    )
    
    # Test detection on TWO different texts
    test_texts = [
        "This is the first test sentence for watermark detection.",
        "Here is a completely different second text to test per-text optimization."
    ]
    
    target_lang = "fra"  # French as target
    
    for i, test_text in enumerate(test_texts, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TEXT {i}: INDEPENDENT PER-TEXT OPTIMIZATION")
        print(f"{'#'*80}")
        
        result = steam_bo.detect_with_bo(test_text, target_lang)
        
        print(f"\nFinal result for text {i}:")
        print(f"  Success: {result.success}")
        print(f"  Best z-score: {result.z_score:.4f}")
        print(f"  Best language: {result.best_intermediate_lang}")
        print(f"  Total evaluations: {result.total_evaluations}")
        
        if result.error:
            print(f"  Error: {result.error}")


if __name__ == "__main__":
    test_pertext_detector()