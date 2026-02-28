#!/usr/bin/env python3
"""
STEAM BO Detector - Per-Text Bayesian Optimization for Pivot Language Selection

This implements the correct STEAM BO approach:
1. For each individual text, run separate BO optimization
2. Find optimal pivot language that maximizes normalized z-score
3. Translation flow: tgt_lang → pivot_lang (single step)
4. Normalization: raw_z_score - avg_validation_z_score_for_pivot_lang

Author: Asim
"""

import os
import json
import numpy as np
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

from skopt import gp_minimize
from skopt.space import Real

# Import your existing components
from genetic_diversity_selector import GeneticDiversitySelector
from realtime_backtranslation import RealtimeBacktranslator
from uriel_genetic_distance import URIELGeneticDistance
from language_code_converter import iso3_to_iso1, iso1_to_iso3, is_valid_iso3
from utils import read_jsonl

# Import watermark detectors
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src_watermark.xsir.watermark import (
    WatermarkWindow as XSIRWindow,
    WatermarkContext as XSIRContext,
)
from src_watermark.kgw.extended_watermark_processor import (
    WatermarkDetector as KGWDetector
)
from src_watermark.uw.detect import Detector as UWDetector


def get_watermark_detector(watermark_method: str, base_model: str, **kwargs):
    """Factory function to create watermark detector."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if watermark_method == "kgw":
        return KGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=kwargs.get('gamma', 0.5),
            seeding_scheme=kwargs.get('seeding_scheme', 'simple_1'),
            device=device,
            tokenizer=tokenizer,
            z_threshold=kwargs.get('z_threshold', 4.0),
            normalizers=kwargs.get('normalizers', []),
            ignore_repeated_ngrams=kwargs.get('ignore_repeated_ngrams', True),
        )
    elif watermark_method in ["xsir", "sir"]:
        watermark_type = kwargs.get('watermark_type', 'context')
        if watermark_type == "window":
            return XSIRWindow(
                device,
                kwargs.get('window_size', 5),
                tokenizer
            )
        elif watermark_type == "context":
            return XSIRContext(
                device,
                kwargs.get('chunk_size', 20),
                tokenizer,
                mapping_file=kwargs.get('mapping_file'),
                delta=kwargs.get('delta', 2.0),
                transform_model_path=kwargs.get('transform_model'),
                embedding_model=kwargs.get('embedding_model', 'paraphrase-multilingual-mpnet-base-v2')
            )
        else:
            raise ValueError(f"Unknown watermark type: {watermark_type}")
    elif watermark_method == "uw":
        return UWDetector(
            model_name=base_model,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown watermark method: {watermark_method}")


@dataclass
class TextSTEAMResult:
    """Result for a single text's STEAM BO optimization."""
    text_id: int
    best_pivot_lang: str
    best_normalized_z_score: float
    best_raw_z_score: float
    best_genetic_distance: float
    evaluations: List[Dict[str, Any]]
    total_evaluations: int
    convergence_iteration: int


@dataclass
class STEAMBOResults:
    """Complete STEAM BO results for all texts."""
    target_lang: str
    text_results: List[TextSTEAMResult]
    overall_stats: Dict[str, Any]


class STEAMBODetector:
    """
    Per-Text STEAM BO Detector for optimal pivot language selection.

    For each text:
    1. Initialize with 3 genetically diverse pivot languages
    2. Use BO to find optimal pivot language that maximizes normalized z-score
    3. Apply best pivot to mod/hum/val versions of the same text
    """

    def __init__(self,
                 watermark_detector,
                 target_lang: str,
                 input_dir: str,
                 output_dir: str,
                 n_initial: int = 3,
                 max_evaluations: int = 8,
                 random_state: int = 42):
        """
        Initialize STEAM BO Detector.

        Args:
            watermark_detector: Watermark detector (KGW/XSIR/etc.)
            target_lang: Target language (e.g., 'fr', 'de', 'es')
            input_dir: Directory containing input files
            output_dir: Directory for output results
            n_initial: Number of initial diverse languages
            max_evaluations: Maximum BO evaluations per text
            random_state: Random seed
        """
        self.watermark_detector = watermark_detector
        self.target_lang = target_lang
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_initial = n_initial
        self.max_evaluations = max_evaluations
        self.random_state = random_state

        # Setup logging first
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.diversity_selector = GeneticDiversitySelector(random_seed=random_state)
        self.backtranslator = RealtimeBacktranslator()
        self.genetic_distance = URIELGeneticDistance()

        # Convert target language to ISO-3 for URIEL
        self.target_lang_iso3 = self._normalize_to_iso3(target_lang)

        # Get available pivot languages (exclude target language and filter for Google Translate support)
        all_pivots = [lang for lang in self.diversity_selector.available_languages
                     if lang != self.target_lang_iso3]

        # Filter to only languages supported by Google Translate
        from deep_translator import GoogleTranslator
        google_supported = set(GoogleTranslator().get_supported_languages(as_dict=True).values())

        self.available_pivots = []
        for lang in all_pivots:
            try:
                lang_iso1 = iso3_to_iso1(lang)
                if lang_iso1 in google_supported:
                    self.available_pivots.append(lang)
            except ValueError:
                # Skip languages without ISO-1 equivalents
                continue

        self.logger.info(f"Filtered to {len(self.available_pivots)} Google Translate supported languages from {len(all_pivots)} total")

        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"Initialized STEAM BO Detector")
        self.logger.info(f"Target language: {target_lang} -> {self.target_lang_iso3}")
        self.logger.info(f"Available pivot languages: {len(self.available_pivots)}")

    def _normalize_to_iso3(self, lang_code: str) -> str:
        """Convert language code to ISO-3 format."""
        if is_valid_iso3(lang_code):
            return lang_code

        iso3 = iso1_to_iso3(lang_code)
        if iso3:
            return iso3

        raise ValueError(f"Cannot normalize language code {lang_code} to ISO-3")

    def _select_initial_pivot_languages(self) -> List[str]:
        """Select initial pivot languages using simple random selection for speed."""
        self.logger.info(f"Using fast random selection from {len(self.available_pivots)} available languages...")

        # Simple random selection - bypass genetic diversity for speed
        np.random.seed(self.random_state)
        n_select = min(self.n_initial, len(self.available_pivots))
        selected_pivots = np.random.choice(self.available_pivots, n_select, replace=False).tolist()

        self.logger.info(f"Selected random pivot languages: {selected_pivots}")
        return selected_pivots

    def _get_validation_baseline(self, pivot_lang: str) -> float:
        """
        Get validation baseline z-score for a pivot language.
        If validation z-score file doesn't exist, create it by translating validation texts.
        """
        val_z_score_file = os.path.join(self.input_dir, f"mc4.{self.target_lang}-{pivot_lang}-back.val.z_score.jsonl")

        # Check if validation z-score file already exists
        if os.path.exists(val_z_score_file):
            try:
                val_data = read_jsonl(val_z_score_file)
                z_scores = [item.get('z_score', 0.0) for item in val_data if item.get('z_score') is not None]

                if z_scores:
                    avg_z_score = sum(z_scores) / len(z_scores)
                    self.logger.debug(f"Validation baseline for {pivot_lang}: {avg_z_score:.4f} (from existing file)")
                    return avg_z_score
                else:
                    self.logger.warning(f"No valid z-scores in existing {val_z_score_file}")

            except Exception as e:
                self.logger.error(f"Error reading validation file {val_z_score_file}: {e}")

        # Validation z-score file doesn't exist or is invalid, create it
        self.logger.info(f"Creating validation baseline for {pivot_lang} by translating validation texts")

        # Load validation texts
        val_text_file = os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.val.jsonl")

        if not os.path.exists(val_text_file):
            self.logger.warning(f"Validation text file not found: {val_text_file}")
            return 0.0

        try:
            val_texts = read_jsonl(val_text_file)

            # Randomly sample 50 validation texts for efficiency
            import random
            random.seed(self.random_state)
            n_val_sample = min(50, len(val_texts))
            val_sample = random.sample(val_texts, n_val_sample)

            # Translate validation texts through pivot language and get z-scores
            validation_z_scores = []
            validation_results = []

            self.logger.info(f"Translating {n_val_sample} validation texts (sampled from {len(val_texts)}) through {pivot_lang}")

            for i, val_item in enumerate(val_sample):
                text_content = val_item.get('response', '')

                if not text_content:
                    self.logger.warning(f"Empty validation text at index {i}")
                    continue

                # Translate and detect
                raw_z_score, success = self._translate_and_detect(text_content, pivot_lang)

                if success:
                    validation_z_scores.append(raw_z_score)

                    # Store result for file
                    validation_results.append({
                        'text_id': i,
                        'z_score': raw_z_score,
                        'pivot_lang': pivot_lang,
                        'text_length': len(text_content)
                    })
                else:
                    self.logger.warning(f"Failed to process validation text {i} through {pivot_lang}")
                    validation_results.append({
                        'text_id': i,
                        'z_score': None,
                        'pivot_lang': pivot_lang,
                        'text_length': len(text_content),
                        'error': 'translation_failed'
                    })

            # Calculate baseline
            if validation_z_scores:
                avg_z_score = sum(validation_z_scores) / len(validation_z_scores)
                self.logger.info(f"Computed validation baseline for {pivot_lang}: {avg_z_score:.4f} "
                               f"(from {len(validation_z_scores)}/{n_val_sample} successful translations, sampled from {len(val_texts)} total)")

                # Save validation z-scores to file for future use
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(val_z_score_file), exist_ok=True)

                    with open(val_z_score_file, 'w') as f:
                        for result in validation_results:
                            f.write(json.dumps(result) + '\n')

                    self.logger.debug(f"Saved validation z-scores to {val_z_score_file}")

                except Exception as e:
                    self.logger.warning(f"Failed to save validation z-scores: {e}")

                return avg_z_score
            else:
                self.logger.error(f"No successful validation translations for {pivot_lang}")
                return 0.0

        except Exception as e:
            self.logger.error(f"Error creating validation baseline for {pivot_lang}: {e}")
            return 0.0

    def _translate_and_detect(self, text: str, pivot_lang: str) -> Tuple[float, bool]:
        """
        Translate text to pivot language and detect watermark.

        Args:
            text: Text in target language
            pivot_lang: Pivot language (ISO-3)

        Returns:
            Tuple of (raw_z_score, success)
        """
        try:
            # Convert to ISO-1 for Google Translate
            pivot_iso1 = iso3_to_iso1(pivot_lang)
            target_iso1 = iso3_to_iso1(self.target_lang_iso3)

            if not pivot_iso1 or not target_iso1:
                self.logger.error(f"Language code conversion failed: {pivot_lang} or {self.target_lang_iso3}")
                return 0.0, False

            # Translate: target_lang → pivot_lang
            translated_text = self.backtranslator.translate_text(text, target_iso1, pivot_iso1)

            if translated_text is None:
                self.logger.error(f"Translation failed: {target_iso1} → {pivot_iso1}")
                return 0.0, False

            # Detect watermark
            detection_result = self.watermark_detector.detect(translated_text)
            raw_z_score = detection_result.get('z_score', 0.0)

            return raw_z_score, True

        except Exception as e:
            self.logger.error(f"Error in translate_and_detect for {pivot_lang}: {e}")
            return 0.0, False

    def _evaluate_pivot_language(self, text: str, pivot_lang: str) -> Dict[str, Any]:
        """
        Evaluate a pivot language for a specific text.

        Args:
            text: Text in target language
            pivot_lang: Pivot language to evaluate

        Returns:
            Dictionary with evaluation results
        """
        # Get raw z-score
        raw_z_score, success = self._translate_and_detect(text, pivot_lang)

        if not success:
            return {
                'pivot_lang': pivot_lang,
                'raw_z_score': 0.0,
                'normalized_z_score': 0.0,
                'genetic_distance': 0.0,
                'validation_baseline': 0.0,
                'success': False
            }

        # Get validation baseline
        validation_baseline = self._get_validation_baseline(pivot_lang)

        # Normalize z-score
        normalized_z_score = raw_z_score - validation_baseline

        # Get genetic distance
        try:
            genetic_distance = self.genetic_distance.get_genetic_distance(
                self.target_lang_iso3, pivot_lang
            )
        except Exception as e:
            self.logger.warning(f"Error getting genetic distance for {pivot_lang}: {e}")
            genetic_distance = 3.0  # Default middle value

        return {
            'pivot_lang': pivot_lang,
            'raw_z_score': raw_z_score,
            'normalized_z_score': normalized_z_score,
            'genetic_distance': genetic_distance,
            'validation_baseline': validation_baseline,
            'success': True
        }

    def _bo_suggest_next_pivot(self, evaluations: List[Dict[str, Any]]) -> str:
        """Use BO to suggest next best pivot language."""
        if len(evaluations) < 2:
            # Not enough data for BO, select random
            evaluated_pivots = {eval_result['pivot_lang'] for eval_result in evaluations}
            remaining_pivots = [lang for lang in self.available_pivots if lang not in evaluated_pivots]

            if remaining_pivots:
                np.random.seed(self.random_state)
                return np.random.choice(remaining_pivots)
            else:
                return self.available_pivots[0]  # Fallback

        try:
            # Prepare BO data
            X_samples = []
            y_samples = []

            for eval_result in evaluations:
                if eval_result['success']:
                    X_samples.append([eval_result['genetic_distance']])
                    y_samples.append(eval_result['normalized_z_score'])

            if len(X_samples) < 2:
                # Still not enough successful evaluations
                return np.random.choice(self.available_pivots)

            # Define search space
            search_space = [Real(0.0, 6.0, name='genetic_distance')]

            # Run BO to get next point
            result = gp_minimize(
                func=lambda x: -self._bo_objective(x, X_samples, y_samples),  # Negative for maximization
                dimensions=search_space,
                n_calls=1,
                n_initial_points=0,
                x0=X_samples,
                y0=[-y for y in y_samples],  # Negative for maximization
                acq_func='EI',  # Use string instead of deprecated acquisition_func
                random_state=self.random_state
            )

            # Get suggested genetic distance
            suggested_distance = result.x[0]

            # Find closest available pivot language
            best_pivot = self._find_pivot_at_distance(suggested_distance, evaluations)
            return best_pivot

        except Exception as e:
            self.logger.error(f"BO suggestion failed: {e}")
            # Fallback to random selection
            evaluated_pivots = {eval_result['pivot_lang'] for eval_result in evaluations}
            remaining_pivots = [lang for lang in self.available_pivots if lang not in evaluated_pivots]

            if remaining_pivots:
                return np.random.choice(remaining_pivots)
            else:
                return self.available_pivots[0]

    def _bo_objective(self, x: List[float], X_samples: List[List[float]], y_samples: List[float]) -> float:
        """BO objective function (placeholder - actual modeling done by GP)."""
        return 0.0

    def _find_pivot_at_distance(self, target_distance: float, evaluations: List[Dict[str, Any]]) -> str:
        """Find pivot language closest to target genetic distance."""
        evaluated_pivots = {eval_result['pivot_lang'] for eval_result in evaluations}

        best_pivot = None
        best_diff = float('inf')

        for pivot_lang in self.available_pivots:
            if pivot_lang in evaluated_pivots:
                continue

            try:
                actual_distance = self.genetic_distance.get_genetic_distance(
                    self.target_lang_iso3, pivot_lang
                )
                diff = abs(actual_distance - target_distance)

                if diff < best_diff:
                    best_diff = diff
                    best_pivot = pivot_lang

            except Exception:
                continue

        return best_pivot if best_pivot else self.available_pivots[0]

    def optimize_single_text(self, text: str, text_id: int) -> TextSTEAMResult:
        """
        Run STEAM BO optimization for a single text.

        Args:
            text: Text content in target language
            text_id: Text identifier

        Returns:
            TextSTEAMResult with optimization results
        """
        self.logger.info(f"Starting STEAM BO for text {text_id}")

        evaluations = []

        # Phase 1: Evaluate initial pivot languages
        initial_pivots = self._select_initial_pivot_languages()

        for pivot_lang in initial_pivots:
            eval_result = self._evaluate_pivot_language(text, pivot_lang)
            evaluations.append(eval_result)

            self.logger.info(f"  Initial {pivot_lang}: norm_z={eval_result['normalized_z_score']:.3f}, "
                           f"dist={eval_result['genetic_distance']:.3f}")

        # Find current best
        successful_evals = [e for e in evaluations if e['success']]
        if not successful_evals:
            self.logger.error(f"No successful evaluations for text {text_id}")
            return TextSTEAMResult(
                text_id=text_id,
                best_pivot_lang="eng",  # Fallback
                best_normalized_z_score=0.0,
                best_raw_z_score=0.0,
                best_genetic_distance=0.0,
                evaluations=evaluations,
                total_evaluations=len(evaluations),
                convergence_iteration=0
            )

        best_eval = max(successful_evals, key=lambda x: x['normalized_z_score'])
        convergence_iteration = len(evaluations)

        # Phase 2: BO optimization loop
        for iteration in range(self.max_evaluations - self.n_initial):
            # Get next pivot from BO
            next_pivot = self._bo_suggest_next_pivot(evaluations)

            if not next_pivot:
                break

            self.logger.info(f"  BO iteration {iteration + 1}: trying {next_pivot}")

            # Evaluate
            eval_result = self._evaluate_pivot_language(text, next_pivot)
            evaluations.append(eval_result)

            # Check if new best
            if eval_result['success'] and eval_result['normalized_z_score'] > best_eval['normalized_z_score']:
                best_eval = eval_result
                convergence_iteration = len(evaluations)

            self.logger.info(f"    {next_pivot}: norm_z={eval_result['normalized_z_score']:.3f}, "
                           f"dist={eval_result['genetic_distance']:.3f}")

        return TextSTEAMResult(
            text_id=text_id,
            best_pivot_lang=best_eval['pivot_lang'],
            best_normalized_z_score=best_eval['normalized_z_score'],
            best_raw_z_score=best_eval['raw_z_score'],
            best_genetic_distance=best_eval['genetic_distance'],
            evaluations=evaluations,
            total_evaluations=len(evaluations),
            convergence_iteration=convergence_iteration
        )

    def run_steam_bo_optimization(self, num_texts: int = 500) -> STEAMBOResults:
        """
        Run STEAM BO optimization for all texts.

        Args:
            num_texts: Number of texts to process

        Returns:
            STEAMBOResults with all optimization results
        """
        self.logger.info(f"Starting STEAM BO optimization for {num_texts} texts")

        # Load input files
        mod_file = os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.mod.jsonl")
        hum_file = os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.hum.jsonl")
        val_file = os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.val.jsonl")

        if not all(os.path.exists(f) for f in [mod_file, hum_file, val_file]):
            raise FileNotFoundError(f"Input files not found in {self.input_dir}")

        mod_data = read_jsonl(mod_file)[:num_texts]
        hum_data = read_jsonl(hum_file)[:num_texts]
        val_data = read_jsonl(val_file)[:num_texts]

        text_results = []

        # Process each text
        for i in range(len(mod_data)):
            try:
                # Use watermarked text for BO optimization
                text_content = mod_data[i].get('response', '')

                if not text_content:
                    self.logger.warning(f"Empty text at index {i}, skipping")
                    continue

                # Run BO optimization for this text
                result = self.optimize_single_text(text_content, i)
                text_results.append(result)

                self.logger.info(f"Text {i}: Best pivot = {result.best_pivot_lang}, "
                               f"Best norm_z = {result.best_normalized_z_score:.3f}")

            except Exception as e:
                self.logger.error(f"Error processing text {i}: {e}")
                continue

        # Compute overall statistics
        if text_results:
            best_scores = [r.best_normalized_z_score for r in text_results]
            pivot_counts = {}
            for r in text_results:
                pivot_counts[r.best_pivot_lang] = pivot_counts.get(r.best_pivot_lang, 0) + 1

            overall_stats = {
                'num_texts_processed': len(text_results),
                'mean_best_score': np.mean(best_scores),
                'std_best_score': np.std(best_scores),
                'max_best_score': np.max(best_scores),
                'min_best_score': np.min(best_scores),
                'pivot_language_distribution': pivot_counts,
                'avg_evaluations_per_text': np.mean([r.total_evaluations for r in text_results])
            }
        else:
            overall_stats = {}

        return STEAMBOResults(
            target_lang=self.target_lang,
            text_results=text_results,
            overall_stats=overall_stats
        )

    def save_results(self, results: STEAMBOResults) -> None:
        """Save STEAM BO results to files."""
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"steam_bo_results_{self.target_lang}.json")

        results_data = {
            'target_lang': results.target_lang,
            'overall_stats': results.overall_stats,
            'text_results': []
        }

        for text_result in results.text_results:
            results_data['text_results'].append({
                'text_id': text_result.text_id,
                'best_pivot_lang': text_result.best_pivot_lang,
                'best_normalized_z_score': text_result.best_normalized_z_score,
                'best_raw_z_score': text_result.best_raw_z_score,
                'best_genetic_distance': text_result.best_genetic_distance,
                'total_evaluations': text_result.total_evaluations,
                'convergence_iteration': text_result.convergence_iteration,
                'evaluations': text_result.evaluations
            })

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")

        # Save summary
        summary_file = os.path.join(self.output_dir, f"steam_bo_summary_{self.target_lang}.json")

        summary_data = {
            'target_lang': results.target_lang,
            'overall_stats': results.overall_stats,
            'best_pivots_per_text': [
                {
                    'text_id': r.text_id,
                    'best_pivot': r.best_pivot_lang,
                    'best_score': r.best_normalized_z_score
                }
                for r in results.text_results
            ]
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        self.logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="STEAM BO Detector")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--watermark_method", type=str, default="kgw", help="Watermark method")
    parser.add_argument("--n_initial", type=int, default=3, help="Initial pivot languages")
    parser.add_argument("--max_evaluations", type=int, default=8, help="Max BO evaluations")
    parser.add_argument("--num_texts", type=int, default=500, help="Number of texts to process")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    # Watermark method specific arguments
    parser.add_argument("--transform_model", type=str, help="Transform model for XSIR/SIR")
    parser.add_argument("--embedding_model", type=str, help="Embedding model for XSIR/SIR")
    parser.add_argument("--mapping_file", type=str, help="Mapping file for XSIR/SIR")

    args = parser.parse_args()

    # Initialize watermark detector
    detector_args = {
        'watermark_method': args.watermark_method,
        'base_model': args.base_model
    }

    if args.transform_model:
        detector_args['transform_model'] = args.transform_model
    if args.embedding_model:
        detector_args['embedding_model'] = args.embedding_model
    if args.mapping_file:
        detector_args['mapping_file'] = args.mapping_file

    watermark_detector = get_watermark_detector(**detector_args)

    # Initialize STEAM BO detector
    steam_detector = STEAMBODetector(
        watermark_detector=watermark_detector,
        target_lang=args.tgt_lang,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_initial=args.n_initial,
        max_evaluations=args.max_evaluations,
        random_state=args.random_state
    )

    # Run optimization
    results = steam_detector.run_steam_bo_optimization(num_texts=args.num_texts)

    # Save results
    steam_detector.save_results(results)

    # Print summary
    print("\n" + "="*50)
    print("STEAM BO Optimization Complete")
    print("="*50)
    print(f"Target Language: {results.target_lang}")
    print(f"Texts Processed: {results.overall_stats.get('num_texts_processed', 0)}")
    print(f"Mean Best Score: {results.overall_stats.get('mean_best_score', 0.0):.4f}")
    print(f"Std Best Score: {results.overall_stats.get('std_best_score', 0.0):.4f}")
    print(f"Avg Evaluations per Text: {results.overall_stats.get('avg_evaluations_per_text', 0.0):.1f}")

    pivot_dist = results.overall_stats.get('pivot_language_distribution', {})
    print(f"\nPivot Language Distribution:")
    for lang, count in sorted(pivot_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count} texts ({count/len(results.text_results)*100:.1f}%)")


if __name__ == "__main__":
    main()