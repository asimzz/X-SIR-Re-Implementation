#!/usr/bin/env python3
"""
STEAM BO Batch Detector: Real Supervisor's Approach

Batch processing of ALL 500 samples:
1. Translate ALL 500 mod/val samples to each language candidate
2. Detect watermarks on ALL translated files (get 500 z-scores per language)
3. Compute AUC performance for each language (500 mod vs 500 val z-scores)
4. BO optimizes: genetic_distance → AUC_performance
5. Find best language that maximizes AUC across full dataset

Author: Asim (implementing REAL supervisor's approach)
"""

import json
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score

from skopt import gp_minimize
from skopt.space import Real

from realtime_backtranslation import RealtimeBacktranslator
from uriel_genetic_distance import URIELGeneticDistance


@dataclass
class BatchSTEAMResult:
    """Result of batch STEAM BO optimization."""
    target_lang: str
    best_intermediate_lang: str
    best_auc: float
    best_genetic_distance: float
    total_evaluations: int
    evaluation_history: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None


class STEAMBOBatchDetector:
    """
    STEAM Batch BO Detector - Real Supervisor's Approach.

    Processes ALL 500 samples at once for each language candidate.
    Optimizes AUC performance across the full dataset.
    """

    def __init__(self,
                 watermark_detector,
                 languages_file: str = 'supported_languages.txt',
                 n_initial: int = 3,
                 max_evaluations: int = 8,
                 acquisition_func: str = 'EI',
                 random_state: int = 42):
        """
        Initialize batch STEAM BO detector.

        Args:
            watermark_detector: Watermark detector instance (XSIR/KGW/UW)
            languages_file: File containing supported language codes
            n_initial: Number of random initial language evaluations
            max_evaluations: Maximum total language evaluations
            acquisition_func: BO acquisition function ('EI', 'UCB')
            random_state: Random seed for reproducibility
        """
        self.watermark_detector = watermark_detector
        self.n_initial = n_initial
        self.max_evaluations = max_evaluations
        self.acquisition_func = acquisition_func
        self.random_state = random_state

        # Load supported languages
        self.all_languages = self._load_languages(languages_file)
        print(f"Loaded {len(self.all_languages)} languages for batch STEAM BO")

        # Initialize components
        self.backtranslator = RealtimeBacktranslator()
        self.genetic_distance_calc = URIELGeneticDistance()

        # Genetic distance bounds for BO
        self.min_genetic_distance = 0.0
        self.max_genetic_distance = 6.0

        # Cache for distance calculations and language mappings
        self.distance_cache = {}
        self.language_distance_map = {}

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
            return ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'jpn', 'kor', 'hin']

    def load_samples(self, mod_file: str, val_file: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load watermarked and validation samples.

        Args:
            mod_file: Path to watermarked samples (mc4.en-fa.mod.z_score.jsonl)
            val_file: Path to validation samples (mc4.en-fa.val.z_score.jsonl)

        Returns:
            Tuple of (mod_samples, val_samples)
        """
        def load_jsonl(file_path: str) -> List[Dict]:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data

        mod_samples = load_jsonl(mod_file)
        val_samples = load_jsonl(val_file)

        self.logger.info(f"Loaded {len(mod_samples)} watermarked samples from {mod_file}")
        self.logger.info(f"Loaded {len(val_samples)} validation samples from {val_file}")

        return mod_samples, val_samples

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
            self.language_distance_map[lang] = distance

        print(f"Built language-distance map for {len(available_langs)} languages")
        print(f"Distance range: [{min(self.language_distance_map.values()):.2f}, {max(self.language_distance_map.values()):.2f}]")

    def distance_to_language(self, target_distance: float, target_lang: str) -> str:
        """Map BO-suggested genetic distance to closest available language."""
        if not self.language_distance_map:
            self.build_language_distance_map(target_lang)

        best_lang = None
        best_diff = float('inf')

        for lang, lang_distance in self.language_distance_map.items():
            diff = abs(lang_distance - target_distance)
            if diff < best_diff:
                best_diff = diff
                best_lang = lang

        return best_lang

    def translate_all_samples(self, samples: List[Dict], target_lang: str,
                             intermediate_lang: str, file_suffix: str) -> List[Dict]:
        """
        Translate ALL samples (500) to intermediate language.

        Args:
            samples: List of all samples to translate
            target_lang: Target language code
            intermediate_lang: Intermediate language for backtranslation
            file_suffix: 'mod' or 'val'

        Returns:
            List of translated samples
        """
        self.logger.info(f"Translating ALL {len(samples)} {file_suffix} samples: {target_lang} → {intermediate_lang}")

        translated_samples = []
        failed_count = 0

        for i, sample in enumerate(samples):
            if i % 100 == 0:
                self.logger.info(f"  Translation progress: {i}/{len(samples)}")

            # Get text from sample
            text = sample.get('response', sample.get('text', sample.get('translation', '')))

            if not text:
                failed_count += 1
                continue

            # Perform backtranslation
            result = self.backtranslator.translate_and_detect(
                text, target_lang, intermediate_lang, self.watermark_detector
            )

            if result['success']:
                translated_sample = sample.copy()
                translated_sample.update({
                    'translated_text': result['translation_result']['translated_text'],
                    'intermediate_lang': intermediate_lang,
                    'z_score': result['z_score'],
                    'translation_success': True
                })
                translated_samples.append(translated_sample)
            else:
                failed_count += 1

        self.logger.info(f"Translation complete: {len(translated_samples)}/{len(samples)} successful, {failed_count} failed")
        return translated_samples

    def compute_auc_performance(self, mod_translated: List[Dict], val_translated: List[Dict]) -> float:
        """
        Compute AUC using validation-normalized z-scores (like evaluate_normalized_detection.py).

        Args:
            mod_translated: Translated watermarked samples with z-scores
            val_translated: Translated validation samples with z-scores

        Returns:
            AUC score (0.0 to 1.0)
        """
        # Extract raw z-scores
        mod_z_scores = [s['z_score'] for s in mod_translated if 'z_score' in s]
        val_z_scores = [s['z_score'] for s in val_translated if 'z_score' in s]

        if len(mod_z_scores) == 0 or len(val_z_scores) == 0:
            self.logger.warning("No z-scores available for AUC computation")
            return 0.0

        # Validation-based normalization (like evaluate_normalized_detection.py)
        # Subtract average validation z-score from both mod and val scores
        avg_val_zscore = np.mean(val_z_scores)

        normalized_mod_scores = [score - avg_val_zscore for score in mod_z_scores]
        normalized_val_scores = [score - avg_val_zscore for score in val_z_scores]

        # Combine normalized z-scores and labels
        all_z_scores = normalized_mod_scores + normalized_val_scores
        labels = [1] * len(normalized_mod_scores) + [0] * len(normalized_val_scores)  # 1=watermarked, 0=validation

        try:
            auc = roc_auc_score(labels, all_z_scores)
            return float(auc)
        except Exception as e:
            self.logger.warning(f"AUC computation failed: {e}")
            return 0.0

    def evaluate_language_batch(self, mod_samples: List[Dict], val_samples: List[Dict],
                               target_lang: str, intermediate_lang: str) -> Dict[str, Any]:
        """
        Evaluate intermediate language on ALL samples (batch processing).

        Args:
            mod_samples: All watermarked samples
            val_samples: All validation samples
            target_lang: Target language code
            intermediate_lang: Intermediate language

        Returns:
            Evaluation result with AUC performance
        """
        genetic_distance = self.get_genetic_distance(target_lang, intermediate_lang)

        # Translate all mod samples
        mod_translated = self.translate_all_samples(mod_samples, target_lang, intermediate_lang, 'mod')

        # Translate all val samples
        val_translated = self.translate_all_samples(val_samples, target_lang, intermediate_lang, 'val')

        # Compute AUC performance
        auc = self.compute_auc_performance(mod_translated, val_translated)

        evaluation_result = {
            'intermediate_lang': intermediate_lang,
            'genetic_distance': genetic_distance,
            'auc': auc,
            'n_mod_samples': len(mod_translated),
            'n_val_samples': len(val_translated),
            'n_total_samples': len(mod_translated) + len(val_translated),
            'success': len(mod_translated) > 0 and len(val_translated) > 0
        }

        return evaluation_result

    def initial_random_sampling(self, mod_samples: List[Dict], val_samples: List[Dict],
                               target_lang: str) -> List[Dict[str, Any]]:
        """
        Perform initial random sampling with batch processing.

        Args:
            mod_samples: All watermarked samples
            val_samples: All validation samples
            target_lang: Target language code

        Returns:
            List of evaluation results
        """
        # Build distance map if not exists
        if not self.language_distance_map:
            self.build_language_distance_map(target_lang)

        # Random sampling of languages
        available_langs = [lang for lang in self.all_languages if lang != target_lang]
        rng = np.random.RandomState(self.random_state)

        n_to_sample = min(self.n_initial, len(available_langs))
        sampled_languages = rng.choice(available_langs, size=n_to_sample, replace=False)

        print(f"\n=== Initial Random Sampling ({n_to_sample} languages) ===")
        print(f"Processing {len(mod_samples)} mod + {len(val_samples)} val = {len(mod_samples) + len(val_samples)} total samples per language")

        evaluation_history = []
        for i, lang in enumerate(sampled_languages, 1):
            print(f"\n[{i}/{n_to_sample}] Evaluating {lang}...")

            result = self.evaluate_language_batch(mod_samples, val_samples, target_lang, lang)
            result['iteration'] = len(evaluation_history) + 1

            evaluation_history.append(result)

            print(f"  AUC: {result['auc']:.4f}, genetic_dist: {result['genetic_distance']:.3f}")
            print(f"  Samples: {result['n_mod_samples']} mod + {result['n_val_samples']} val = {result['n_total_samples']} total")

        return evaluation_history

    def create_bo_objective(self, mod_samples: List[Dict], val_samples: List[Dict],
                           target_lang: str, evaluation_history: List[Dict[str, Any]]) -> Tuple:
        """
        Create BO objective function optimizing AUC in continuous genetic distance space.

        Args:
            mod_samples: All watermarked samples
            val_samples: All validation samples
            target_lang: Target language
            evaluation_history: Previous evaluations

        Returns:
            Tuple of (objective_function, dimensions)
        """
        dimensions = [Real(self.min_genetic_distance, self.max_genetic_distance, name='genetic_distance')]

        def objective_function(x):
            """
            BO objective: maximize AUC by optimizing genetic distance.

            Args:
                x: [genetic_distance] suggested by BO

            Returns:
                Negative AUC (BO minimizes)
            """
            target_distance = x[0]
            intermediate_lang = self.distance_to_language(target_distance, target_lang)

            # Check if this language was already evaluated
            for record in evaluation_history:
                if record['intermediate_lang'] == intermediate_lang:
                    return -record['auc']  # Return cached result

            # Evaluate new language on ALL samples
            print(f"\n  BO iteration {len(evaluation_history) + 1}: Distance {target_distance:.3f} → {intermediate_lang}")

            result = self.evaluate_language_batch(mod_samples, val_samples, target_lang, intermediate_lang)
            result['iteration'] = len(evaluation_history) + 1
            result['suggested_distance'] = target_distance

            evaluation_history.append(result)

            print(f"  AUC: {result['auc']:.4f}")

            if result['success']:
                return -result['auc']  # Negative for minimization
            else:
                return 1.0  # Large penalty for failed evaluations

        return objective_function, dimensions

    def optimize_with_batch_bo(self, mod_file: str, val_file: str, target_lang: str) -> BatchSTEAMResult:
        """
        Perform batch STEAM BO optimization.

        Args:
            mod_file: Path to watermarked samples file
            val_file: Path to validation samples file
            target_lang: Target language code

        Returns:
            BatchSTEAMResult with best AUC and intermediate language
        """
        print(f"\n{'='*80}")
        print(f"STEAM Batch BO Optimization (Real Supervisor's Approach)")
        print(f"{'='*80}")
        print(f"Watermarked file: {mod_file}")
        print(f"Validation file: {val_file}")
        print(f"Target language: {target_lang}")
        print(f"Max evaluations: {self.max_evaluations}")

        try:
            # Step 1: Load all samples
            mod_samples, val_samples = self.load_samples(mod_file, val_file)

            # Step 2: Initial random sampling (batch processing)
            evaluation_history = self.initial_random_sampling(mod_samples, val_samples, target_lang)

            if not evaluation_history:
                return BatchSTEAMResult(
                    target_lang=target_lang,
                    best_intermediate_lang=None,
                    best_auc=0.0,
                    best_genetic_distance=0.0,
                    total_evaluations=0,
                    evaluation_history=[],
                    success=False,
                    error="No initial evaluations completed"
                )

            # Step 3: BO optimization loop
            n_bo_calls = max(0, self.max_evaluations - len(evaluation_history))

            if n_bo_calls > 0:
                print(f"\n=== Bayesian Optimization ({n_bo_calls} additional evaluations) ===")

                objective_func, dimensions = self.create_bo_objective(
                    mod_samples, val_samples, target_lang, evaluation_history)

                successful_evals = [e for e in evaluation_history if e['success']]

                if len(successful_evals) > 0:
                    x0 = [[e['genetic_distance']] for e in successful_evals]
                    y0 = [-e['auc'] for e in successful_evals]

                    print(f"Providing {len(x0)} initial observations to BO")

                    try:
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

            # Step 4: Find best result
            successful_evals = [e for e in evaluation_history if e['success']]

            if not successful_evals:
                return BatchSTEAMResult(
                    target_lang=target_lang,
                    best_intermediate_lang=None,
                    best_auc=0.0,
                    best_genetic_distance=0.0,
                    total_evaluations=len(evaluation_history),
                    evaluation_history=evaluation_history,
                    success=False,
                    error="No successful evaluations"
                )

            # Find best AUC
            best_result = max(successful_evals, key=lambda x: x['auc'])

            print(f"\n{'='*80}")
            print(f"BEST RESULT (Batch Processing)")
            print(f"{'='*80}")
            print(f"Best intermediate language: {best_result['intermediate_lang']}")
            print(f"Best AUC: {best_result['auc']:.4f}")
            print(f"Genetic distance: {best_result['genetic_distance']:.3f}")
            print(f"Total evaluations: {len(evaluation_history)}")
            print(f"Samples per evaluation: {best_result['n_total_samples']}")

            return BatchSTEAMResult(
                target_lang=target_lang,
                best_intermediate_lang=best_result['intermediate_lang'],
                best_auc=best_result['auc'],
                best_genetic_distance=best_result['genetic_distance'],
                total_evaluations=len(evaluation_history),
                evaluation_history=evaluation_history,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Batch STEAM BO optimization failed: {e}")
            import traceback
            traceback.print_exc()

            return BatchSTEAMResult(
                target_lang=target_lang,
                best_intermediate_lang=None,
                best_auc=0.0,
                best_genetic_distance=0.0,
                total_evaluations=0,
                evaluation_history=[],
                success=False,
                error=str(e)
            )


def test_batch_steam_detector():
    """Test the batch STEAM BO detector."""
    # Mock watermark detector
    class MockWatermarkDetector:
        def detect(self, text):
            # Mock different z-scores for watermarked vs human text
            base_score = np.random.normal(3.0, 1.0)  # Biased toward detection
            return {"z_score": base_score, "biases": []}

    # Create mock sample files
    def create_mock_samples(n_samples: int, is_watermarked: bool = True) -> List[Dict]:
        samples = []
        for i in range(n_samples):
            sample = {
                'id': i,
                'text': f"This is {'watermarked' if is_watermarked else 'human'} sample {i}.",
                'response': f"Sample {i} content for testing."
            }
            samples.append(sample)
        return samples

    # Initialize detector
    mock_detector = MockWatermarkDetector()
    batch_steam = STEAMBOBatchDetector(
        watermark_detector=mock_detector,
        n_initial=2,
        max_evaluations=4
    )

    # Create mock data (small for testing)
    mod_samples = create_mock_samples(10, is_watermarked=True)   # Mock 10 watermarked samples
    val_samples = create_mock_samples(10, is_watermarked=False)  # Mock 10 human samples

    # Save to temporary files
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as mod_file:
        for sample in mod_samples:
            mod_file.write(json.dumps(sample) + '\n')
        mod_file_path = mod_file.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as val_file:
        for sample in val_samples:
            val_file.write(json.dumps(sample) + '\n')
        val_file_path = val_file.name

    try:
        print(f"\n{'#'*80}")
        print(f"# TESTING BATCH STEAM BO WITH SUPERVISOR'S APPROACH")
        print(f"{'#'*80}")

        # Test batch optimization
        result = batch_steam.optimize_with_batch_bo(
            mod_file=mod_file_path,
            val_file=val_file_path,
            target_lang="fra"
        )

        print(f"\nFinal batch result:")
        print(f"  Success: {result.success}")
        print(f"  Best AUC: {result.best_auc:.4f}")
        print(f"  Best language: {result.best_intermediate_lang}")
        print(f"  Genetic distance: {result.best_genetic_distance:.3f}")
        print(f"  Total evaluations: {result.total_evaluations}")

        if result.error:
            print(f"  Error: {result.error}")

    finally:
        # Cleanup
        os.unlink(mod_file_path)
        os.unlink(val_file_path)


if __name__ == "__main__":
    test_batch_steam_detector()