#!/usr/bin/env python3
"""
STEAM BO Detector: Supervisor's Correct Approach

Bayesian Optimization with:
- Normalized z-scores
- Continuous genetic distance optimization
- Distance-to-language mapping
- Global BO learning per text

Author: Asim (implementing supervisor's approach)
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from skopt import gp_minimize
from skopt.space import Real
from skopt.acquisition import gaussian_ei, gaussian_ucb

from realtime_backtranslation import RealtimeBacktranslator
from uriel_genetic_distance import URIELGeneticDistance


@dataclass
class STEAMDetectionResult:
    """Result of STEAM BO-enhanced watermark detection."""
    text: str
    target_lang: str
    normalized_z_score: float
    raw_z_score: float
    best_intermediate_lang: str
    best_genetic_distance: float
    total_evaluations: int
    evaluation_history: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None


class STEAMBODetector:
    """
    STEAM Bayesian Optimization Enhanced Watermark Detector.

    Supervisor's approach:
    1. Random initial languages → normalized z-scores
    2. BO in continuous genetic distance space
    3. BO suggests distances → map to languages
    4. Accumulate BO knowledge per text
    """

    def __init__(self,
                 watermark_detector,
                 languages_file: str = 'supported_languages.txt',
                 n_initial: int = 3,
                 max_evaluations: int = 8,
                 acquisition_func: str = 'EI',
                 random_state: int = 42,
                 normalization_samples: int = 100):
        """
        Initialize STEAM BO detector with supervisor's approach.

        Args:
            watermark_detector: Watermark detector instance (XSIR/KGW/UW)
            languages_file: File containing supported language codes
            n_initial: Number of random initial language evaluations
            max_evaluations: Maximum total language evaluations per text
            acquisition_func: BO acquisition function ('EI', 'UCB')
            random_state: Random seed for reproducibility
            normalization_samples: Number of samples for z-score calibration
        """
        self.watermark_detector = watermark_detector
        self.n_initial = n_initial
        self.max_evaluations = max_evaluations
        self.acquisition_func = acquisition_func
        self.random_state = random_state
        self.normalization_samples = normalization_samples

        # Load supported languages
        self.all_languages = self._load_languages(languages_file)
        print(f"Loaded {len(self.all_languages)} deep_translator-supported languages for STEAM BO")

        # Initialize components
        self.backtranslator = RealtimeBacktranslator()
        self.genetic_distance_calc = URIELGeneticDistance()

        # Z-score normalization parameters (will be calibrated)
        self.z_score_mean = 0.0
        self.z_score_std = 1.0
        self.is_calibrated = False

        # Genetic distance bounds for BO
        self.min_genetic_distance = 0.0
        self.max_genetic_distance = 6.0  # Typical URIEL range

        # Cache for distance calculations and language mappings
        self.distance_cache = {}
        self.language_distance_map = {}  # lang -> distance from target

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_languages(self, languages_file: str) -> List[str]:
        """Load supported language codes from file."""
        try:
            with open(languages_file, 'r') as f:
                languages = [line.strip() for line in f if line.strip()]
            return sorted(languages)
        except Exception as e:
            print(f"Error loading languages from {languages_file}: {e}")
            # Fallback to common supported languages
            return ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'jpn', 'kor', 'hin']

    def calibrate_z_score_normalization(self, sample_texts: List[str], target_lang: str):
        """
        Calibrate z-score normalization using sample texts.

        Args:
            sample_texts: List of texts for calibration
            target_lang: Target language code
        """
        print(f"Calibrating z-score normalization with {len(sample_texts)} texts...")

        z_scores = []
        calibration_samples = min(len(sample_texts), self.normalization_samples)

        # Sample a few languages for calibration
        available_langs = [lang for lang in self.all_languages if lang != target_lang]
        rng = np.random.RandomState(self.random_state)
        calibration_languages = rng.choice(available_langs, size=min(5, len(available_langs)), replace=False)

        for i, text in enumerate(sample_texts[:calibration_samples]):
            if i % 20 == 0:
                print(f"  Calibration progress: {i}/{calibration_samples}")

            for lang in calibration_languages:
                result = self.backtranslator.translate_and_detect(
                    text, target_lang, lang, self.watermark_detector
                )
                if result['success']:
                    z_scores.append(result['z_score'])

        if z_scores:
            self.z_score_mean = np.mean(z_scores)
            self.z_score_std = np.std(z_scores)
            self.is_calibrated = True

            print(f"Z-score calibration complete:")
            print(f"  Mean: {self.z_score_mean:.4f}")
            print(f"  Std:  {self.z_score_std:.4f}")
            print(f"  Samples: {len(z_scores)}")
        else:
            print("WARNING: Z-score calibration failed, using defaults (mean=0, std=1)")
            self.z_score_mean = 0.0
            self.z_score_std = 1.0
            self.is_calibrated = False

    def normalize_z_score(self, raw_z_score: float) -> float:
        """
        Normalize z-score using calibrated parameters.

        Args:
            raw_z_score: Raw z-score from watermark detector

        Returns:
            Normalized z-score
        """
        if self.z_score_std == 0:
            return 0.0
        return (raw_z_score - self.z_score_mean) / self.z_score_std

    def get_genetic_distance(self, lang1: str, lang2: str) -> float:
        """Get cached genetic distance between languages."""
        cache_key = tuple(sorted([lang1, lang2]))
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        distance = self.genetic_distance_calc.get_genetic_distance(lang1, lang2)
        self.distance_cache[cache_key] = distance
        return distance

    def build_language_distance_map(self, target_lang: str):
        """Build mapping from genetic distances to languages."""
        self.language_distance_map = {}

        available_langs = [lang for lang in self.all_languages if lang != target_lang]

        for lang in available_langs:
            distance = self.get_genetic_distance(target_lang, lang)
            # Store language with its distance (for mapping back from BO suggestions)
            self.language_distance_map[lang] = distance

        print(f"Built language-distance map for {len(available_langs)} languages")
        print(f"Distance range: [{min(self.language_distance_map.values()):.2f}, {max(self.language_distance_map.values()):.2f}]")

    def distance_to_language(self, target_distance: float, target_lang: str) -> str:
        """
        Map BO-suggested genetic distance to closest available language.

        Args:
            target_distance: Distance suggested by BO
            target_lang: Target language code

        Returns:
            Language code closest to the target distance
        """
        if not self.language_distance_map:
            self.build_language_distance_map(target_lang)

        # Find language with distance closest to BO suggestion
        best_lang = None
        best_diff = float('inf')

        for lang, lang_distance in self.language_distance_map.items():
            diff = abs(lang_distance - target_distance)
            if diff < best_diff:
                best_diff = diff
                best_lang = lang

        return best_lang

    def evaluate_language(self, text: str, target_lang: str, intermediate_lang: str) -> Dict[str, Any]:
        """
        Evaluate watermark detection via intermediate language.

        Args:
            text: Input text
            target_lang: Target language code
            intermediate_lang: Intermediate language for backtranslation

        Returns:
            Evaluation result with normalized z-score
        """
        # Get genetic distance
        genetic_distance = self.get_genetic_distance(target_lang, intermediate_lang)

        # Perform backtranslation and detection
        result = self.backtranslator.translate_and_detect(
            text, target_lang, intermediate_lang, self.watermark_detector
        )

        if result['success']:
            raw_z_score = result['z_score']
            normalized_z_score = self.normalize_z_score(raw_z_score)
        else:
            raw_z_score = -np.inf
            normalized_z_score = -np.inf

        evaluation_result = {
            'intermediate_lang': intermediate_lang,
            'genetic_distance': genetic_distance,
            'raw_z_score': raw_z_score,
            'normalized_z_score': normalized_z_score,
            'success': result['success'],
            'translation_result': result
        }

        return evaluation_result

    def initial_random_sampling(self, text: str, target_lang: str) -> List[Dict[str, Any]]:
        """
        Perform initial random sampling in genetic distance space.

        Args:
            text: Input text for detection
            target_lang: Target language code

        Returns:
            List of evaluation results
        """
        # Build distance map if not exists
        if not self.language_distance_map:
            self.build_language_distance_map(target_lang)

        # Random sampling of genetic distances (not languages directly)
        available_langs = [lang for lang in self.all_languages if lang != target_lang]
        rng = np.random.RandomState(self.random_state + hash(text[:50]) % 10000)

        n_to_sample = min(self.n_initial, len(available_langs))
        sampled_languages = rng.choice(available_langs, size=n_to_sample, replace=False)

        print(f"\n=== Initial Random Sampling ({n_to_sample} languages) ===")

        evaluation_history = []
        for i, lang in enumerate(sampled_languages, 1):
            print(f"  [{i}/{n_to_sample}] Evaluating {lang}...", end=' ')

            result = self.evaluate_language(text, target_lang, lang)
            result['iteration'] = len(evaluation_history) + 1

            evaluation_history.append(result)

            print(f"raw_z: {result['raw_z_score']:.4f}, norm_z: {result['normalized_z_score']:.4f}, dist: {result['genetic_distance']:.3f}")

        return evaluation_history

    def create_bo_objective(self, text: str, target_lang: str,
                           evaluation_history: List[Dict[str, Any]]) -> Tuple:
        """
        Create BO objective function optimizing in continuous genetic distance space.

        Args:
            text: Text being evaluated
            target_lang: Target language
            evaluation_history: Previous evaluations

        Returns:
            Tuple of (objective_function, dimensions)
        """
        # BO optimizes in continuous genetic distance space
        dimensions = [Real(self.min_genetic_distance, self.max_genetic_distance, name='genetic_distance')]

        def objective_function(x):
            """
            BO objective: maximize normalized z-score by optimizing genetic distance.

            Args:
                x: [genetic_distance] suggested by BO

            Returns:
                Negative normalized z-score (BO minimizes)
            """
            target_distance = x[0]  # BO-suggested genetic distance

            # Map distance to actual language
            intermediate_lang = self.distance_to_language(target_distance, target_lang)

            # Check if this language was already evaluated
            for record in evaluation_history:
                if record['intermediate_lang'] == intermediate_lang:
                    return -record['normalized_z_score']  # Return cached result

            # Evaluate new language
            print(f"  BO iteration {len(evaluation_history) + 1}: Distance {target_distance:.3f} → {intermediate_lang}...", end=' ')

            result = self.evaluate_language(text, target_lang, intermediate_lang)
            result['iteration'] = len(evaluation_history) + 1
            result['suggested_distance'] = target_distance

            evaluation_history.append(result)

            print(f"norm_z: {result['normalized_z_score']:.4f}")

            if result['success']:
                return -result['normalized_z_score']  # Negative for minimization
            else:
                return 10.0  # Large penalty for failed evaluations

        return objective_function, dimensions

    def detect_with_steam_bo(self, text: str, target_lang: str) -> STEAMDetectionResult:
        """
        Perform STEAM BO-enhanced watermark detection using supervisor's approach.

        Args:
            text: Input text to check for watermarks
            target_lang: Target language code

        Returns:
            STEAMDetectionResult with best normalized z-score and intermediate language
        """
        print(f"\n{'='*70}")
        print(f"STEAM BO Detection (Supervisor's Approach)")
        print(f"{'='*70}")
        print(f"Text: {text[:100]}...")
        print(f"Target language: {target_lang}")
        print(f"Max evaluations: {self.max_evaluations}")
        print(f"Calibrated: {self.is_calibrated}")

        try:
            # Step 1: Initial random sampling
            evaluation_history = self.initial_random_sampling(text, target_lang)

            if not evaluation_history:
                return STEAMDetectionResult(
                    text=text,
                    target_lang=target_lang,
                    normalized_z_score=-np.inf,
                    raw_z_score=-np.inf,
                    best_intermediate_lang=None,
                    best_genetic_distance=0.0,
                    total_evaluations=0,
                    evaluation_history=[],
                    success=False,
                    error="No initial evaluations completed"
                )

            # Step 2: BO optimization in continuous genetic distance space
            n_bo_calls = max(0, self.max_evaluations - len(evaluation_history))

            if n_bo_calls > 0:
                print(f"\n=== Bayesian Optimization ({n_bo_calls} additional evaluations) ===")

                # Create BO objective in genetic distance space
                objective_func, dimensions = self.create_bo_objective(text, target_lang, evaluation_history)

                # Prepare initial observations for BO
                successful_evals = [e for e in evaluation_history if e['success']]

                if len(successful_evals) > 0:
                    # Format initial observations (distance, normalized_z_score)
                    x0 = [[e['genetic_distance']] for e in successful_evals]
                    y0 = [-e['normalized_z_score'] for e in successful_evals]  # Negative for minimization

                    print(f"Providing {len(x0)} initial observations to BO")

                    try:
                        # Run BO in continuous genetic distance space
                        result = gp_minimize(
                            func=objective_func,
                            dimensions=dimensions,
                            n_calls=n_bo_calls,
                            x0=x0,
                            y0=y0,
                            acq_func=self.acquisition_func.lower(),
                            random_state=self.random_state,
                            verbose=False
                        )
                        print(f"BO optimization completed successfully")

                    except Exception as e:
                        self.logger.error(f"BO optimization failed: {e}")
                else:
                    print("WARNING: No successful initial evaluations for BO")

            # Step 3: Find best result
            successful_evals = [e for e in evaluation_history if e['success']]

            if not successful_evals:
                return STEAMDetectionResult(
                    text=text,
                    target_lang=target_lang,
                    normalized_z_score=-np.inf,
                    raw_z_score=-np.inf,
                    best_intermediate_lang=None,
                    best_genetic_distance=0.0,
                    total_evaluations=len(evaluation_history),
                    evaluation_history=evaluation_history,
                    success=False,
                    error="No successful evaluations"
                )

            # Find best normalized z-score
            best_result = max(successful_evals, key=lambda x: x['normalized_z_score'])

            print(f"\n{'='*70}")
            print(f"BEST RESULT")
            print(f"{'='*70}")
            print(f"Best intermediate language: {best_result['intermediate_lang']}")
            print(f"Raw z-score: {best_result['raw_z_score']:.4f}")
            print(f"Normalized z-score: {best_result['normalized_z_score']:.4f}")
            print(f"Genetic distance: {best_result['genetic_distance']:.3f}")
            print(f"Total evaluations: {len(evaluation_history)}")

            return STEAMDetectionResult(
                text=text,
                target_lang=target_lang,
                normalized_z_score=best_result['normalized_z_score'],
                raw_z_score=best_result['raw_z_score'],
                best_intermediate_lang=best_result['intermediate_lang'],
                best_genetic_distance=best_result['genetic_distance'],
                total_evaluations=len(evaluation_history),
                evaluation_history=evaluation_history,
                success=True
            )

        except Exception as e:
            self.logger.error(f"STEAM BO detection failed: {e}")
            import traceback
            traceback.print_exc()

            return STEAMDetectionResult(
                text=text,
                target_lang=target_lang,
                normalized_z_score=-np.inf,
                raw_z_score=-np.inf,
                best_intermediate_lang=None,
                best_genetic_distance=0.0,
                total_evaluations=0,
                evaluation_history=[],
                success=False,
                error=str(e)
            )


def test_steam_bo_detector():
    """Test the corrected STEAM BO detector."""
    # Mock watermark detector
    class MockWatermarkDetector:
        def detect(self, text):
            # Mock z-scores with some variation
            base_score = np.random.normal(2.0, 1.5)  # Typical watermark scores
            return {"z_score": base_score, "biases": []}

    # Sample texts for calibration
    calibration_texts = [
        "This is a calibration text sample.",
        "Another text for z-score normalization.",
        "Watermark detection calibration sample.",
    ]

    # Initialize detector
    mock_detector = MockWatermarkDetector()
    steam_bo = STEAMBODetector(
        watermark_detector=mock_detector,
        n_initial=3,
        max_evaluations=6,
        normalization_samples=10  # Small for testing
    )

    # Calibrate normalization
    target_lang = "fra"
    steam_bo.calibrate_z_score_normalization(calibration_texts, target_lang)

    # Test detection
    test_text = "This is a test sentence for STEAM BO watermark detection with genetic distance optimization."

    print(f"\n{'#'*80}")
    print(f"# TESTING STEAM BO WITH SUPERVISOR'S APPROACH")
    print(f"{'#'*80}")

    result = steam_bo.detect_with_steam_bo(test_text, target_lang)

    print(f"\nFinal result:")
    print(f"  Success: {result.success}")
    print(f"  Raw z-score: {result.raw_z_score:.4f}")
    print(f"  Normalized z-score: {result.normalized_z_score:.4f}")
    print(f"  Best language: {result.best_intermediate_lang}")
    print(f"  Genetic distance: {result.best_genetic_distance:.3f}")
    print(f"  Total evaluations: {result.total_evaluations}")

    if result.error:
        print(f"  Error: {result.error}")


if __name__ == "__main__":
    test_steam_bo_detector()