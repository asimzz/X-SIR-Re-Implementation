#!/usr/bin/env python3
"""
STEAM BO Detector: Bayesian Optimization Enhanced Watermark Detection

This module implements a BO-enhanced version of STEAM that intelligently
selects intermediate languages for backtranslation-based watermark detection.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from skopt import gp_minimize
from skopt.space import Categorical

from realtime_backtranslation import RealtimeBacktranslator
from genetic_distance_evaluator import GeneticDistanceEvaluator


@dataclass
class DetectionResult:
    """Result of BO-enhanced watermark detection."""
    z_score: float
    best_intermediate_lang: str
    total_evaluations: int
    evaluation_history: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None


class SteamBODetector:
    """
    Bayesian Optimization Enhanced STEAM Watermark Detector.

    Uses BO to intelligently select intermediate languages for backtranslation
    instead of exhaustive search across all available languages.
    """

    def __init__(self,
                 watermark_detector,
                 n_initial: int = 2,
                 max_evaluations: int = 6,
                 acquisition_func: str = 'EI',
                 random_state: int = 42):
        """
        Initialize BO-enhanced STEAM detector.

        Args:
            watermark_detector: Watermark detector instance (XSIR/KGW/UW)
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

        # Initialize components
        self.backtranslator = RealtimeBacktranslator()
        self.distance_evaluator = GeneticDistanceEvaluator()

        # All available intermediate languages (from the original STEAM evaluation)
        self.available_languages = [
            "en", "fr", "de", "it", "es", "pt",  # High-resource
            "pl", "nl", "ru", "hi", "ko", "ja",  # Medium-resource
            "bn", "fa", "vi", "iw", "uk", "ta"   # Low-resource
        ]

        # Setup logging
        logging.basicConfig(level=logging.WARNING)  # Minimal logging
        self.logger = logging.getLogger(__name__)

    def get_language_features(self, target_lang: str, intermediate_lang: str) -> np.ndarray:
        """
        Extract continuous features for a language pair.

        Args:
            target_lang: Target language code
            intermediate_lang: Intermediate language code

        Returns:
            Feature vector for BO (continuous features only)
        """
        try:
            # Primary feature: genetic distance
            genetic_distance = self.distance_evaluator.get_genetic_distance(target_lang, intermediate_lang)

            # Additional continuous features
            # Language code length (normalized)
            code_length_feature = len(intermediate_lang) / 10.0

            # Simple hash-based language similarity (crude approximation)
            lang_hash_similarity = abs(hash(target_lang) - hash(intermediate_lang)) / (2**31 - 1)

            # Combine continuous features
            features = np.array([
                genetic_distance,           # Primary: genetic distance [0,1]
                code_length_feature,        # Secondary: code length [0,1]
                lang_hash_similarity,       # Tertiary: simple similarity [0,1]
            ])

            return features

        except Exception as e:
            self.logger.warning(f"Feature extraction failed for {target_lang}-{intermediate_lang}: {e}")
            # Return default features
            return np.array([0.5, 0.3, 0.5])

    def initial_sampling(self, text: str, target_lang: str) -> List[Dict[str, Any]]:
        """
        Perform initial random sampling of intermediate languages.

        Args:
            text: Input text for detection
            target_lang: Target language code

        Returns:
            List of evaluation results
        """

        # Filter out target language
        available_langs = [lang for lang in self.available_languages if lang != target_lang]

        # Simple random sampling
        np.random.seed(self.random_state)
        n_to_sample = min(self.n_initial, len(available_langs))
        sampled_languages = np.random.choice(available_langs, size=n_to_sample, replace=False)

        # Evaluate each sampled language
        evaluation_history = []
        for lang in sampled_languages:
            result = self.backtranslator.translate_and_detect(
                text, target_lang, lang, self.watermark_detector
            )

            evaluation_record = {
                'iteration': len(evaluation_history) + 1,
                'intermediate_lang': lang,
                'z_score': result['z_score'] if result['success'] else -np.inf,
                'success': result['success'],
                'features': self.get_language_features(target_lang, lang),
                'translation_result': result
            }

            evaluation_history.append(evaluation_record)


        return evaluation_history

    def create_bo_objective(self, text: str, target_lang: str,
                           evaluation_history: List[Dict[str, Any]]) -> tuple:
        """
        Create BO objective function and search space.

        Args:
            text: Input text for detection
            target_lang: Target language code
            evaluation_history: Current evaluation history

        Returns:
            Tuple of (objective_function, dimensions)
        """
        # Filter out target language
        available_langs = [lang for lang in self.available_languages if lang != target_lang]

        # Create categorical search space
        dimensions = [Categorical(available_langs, name='intermediate_lang')]

        def objective_function(params):
            """Objective function for BO (returns negative z-score for minimization)."""
            intermediate_lang = params[0]

            # Check if already evaluated
            for record in evaluation_history:
                if record['intermediate_lang'] == intermediate_lang:
                    return -record['z_score'] if record['success'] else 1000.0

            # Evaluate new language
            result = self.backtranslator.translate_and_detect(
                text, target_lang, intermediate_lang, self.watermark_detector
            )

            evaluation_record = {
                'iteration': len(evaluation_history) + 1,
                'intermediate_lang': intermediate_lang,
                'z_score': result['z_score'] if result['success'] else -np.inf,
                'success': result['success'],
                'features': self.get_language_features(target_lang, intermediate_lang),
                'translation_result': result
            }

            evaluation_history.append(evaluation_record)

            if result['success']:
                return -result['z_score']  # Negative for minimization
            else:
                return 1000.0  # Large penalty for failed evaluations

        return objective_function, dimensions

    def detect_with_bo(self, text: str, target_lang: str) -> DetectionResult:
        """
        Perform BO-enhanced watermark detection.

        Args:
            text: Input text to check for watermarks
            target_lang: Target language code

        Returns:
            DetectionResult with best z-score and intermediate language
        """

        try:
            # Step 1: Initial random sampling
            evaluation_history = self.initial_sampling(text, target_lang)

            if not evaluation_history:
                return DetectionResult(
                    z_score=-np.inf,
                    best_intermediate_lang=None,
                    total_evaluations=0,
                    evaluation_history=[],
                    success=False,
                    error="No initial evaluations completed"
                )

            # Step 2: BO optimization loop
            n_bo_calls = max(1, self.max_evaluations - len(evaluation_history))

            if n_bo_calls > 0:
                # Create BO objective and search space
                objective_func, dimensions = self.create_bo_objective(text, target_lang, evaluation_history)

                # Run Bayesian optimization
                try:
                    result = gp_minimize(
                        func=objective_func,
                        dimensions=dimensions,
                        n_calls=n_bo_calls,
                        n_initial_points=min(n_bo_calls, 1),
                        acq_func=self.acquisition_func,
                        random_state=self.random_state,
                        verbose=False
                    )
                except Exception as e:
                    pass

            # Step 3: Find best result
            best_result = None
            best_z_score = -np.inf

            for record in evaluation_history:
                if record['success'] and record['z_score'] > best_z_score:
                    best_z_score = record['z_score']
                    best_result = record

            if best_result is None:
                return DetectionResult(
                    z_score=-np.inf,
                    best_intermediate_lang=None,
                    total_evaluations=len(evaluation_history),
                    evaluation_history=evaluation_history,
                    success=False,
                    error="No successful evaluations"
                )


            return DetectionResult(
                z_score=best_z_score,
                best_intermediate_lang=best_result['intermediate_lang'],
                total_evaluations=len(evaluation_history),
                evaluation_history=evaluation_history,
                success=True
            )

        except Exception as e:
            return DetectionResult(
                z_score=-np.inf,
                best_intermediate_lang=None,
                total_evaluations=0,
                evaluation_history=[],
                success=False,
                error=str(e)
            )


def test_steam_bo_detector():
    """Test function for STEAM BO detector."""
    # Mock watermark detector for testing
    class MockWatermarkDetector:
        def detect(self, text):
            # Return mock z-score based on text length
            z_score = len(text) / 100.0 + np.random.normal(0, 0.1)
            return {"z_score": z_score, "biases": []}

    # Initialize detector
    mock_detector = MockWatermarkDetector()
    steam_bo = SteamBODetector(
        watermark_detector=mock_detector,
        n_initial=2,
        max_evaluations=4
    )

    # Test detection
    test_text = "This is a test sentence for watermark detection using the BO-enhanced STEAM system."
    target_lang = "en"

    print(f"Testing BO-enhanced detection...")
    print(f"Text: {test_text}")
    print(f"Target language: {target_lang}")

    result = steam_bo.detect_with_bo(test_text, target_lang)

    print(f"\nResults:")
    print(f"Success: {result.success}")
    print(f"Best z-score: {result.z_score}")
    print(f"Best intermediate language: {result.best_intermediate_lang}")
    print(f"Total evaluations: {result.total_evaluations}")

    if result.error:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    test_steam_bo_detector()