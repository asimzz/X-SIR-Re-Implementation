#!/usr/bin/env python3
"""
STEAM BO Detector - Per-Text Bayesian Optimization for Pivot Language Selection

This implements the correct STEAM BO approach:
1. For each individual text, run separate BO optimization
2. Find optimal pivot language that maximizes normalized z-score
3. Translation flow: tgt_lang → pivot_lang (single step)
4. Normalization: raw_z_score - avg_validation_z_score_for_pivot_lang

Output: single JSONL file (mc4.{target_lang}.bo.z_score.jsonl) with one entry per text:
  {z_score, prompt, response} where response is text in best pivot language

Author: Asim
"""

import os
import json
import numpy as np
import logging
import argparse
from typing import Dict, List, Any, Tuple

from skopt import gp_minimize
from skopt.space import Real

# Import your existing components
from realtime_backtranslation import RealtimeBacktranslator
from uriel_genetic_distance import URIELGeneticDistance
from language_code_converter import iso3_to_iso1, iso1_to_iso3, is_valid_iso3
from utils import read_jsonl

# Import watermark detector
import torch
from transformers import AutoTokenizer
from src_watermark.kgw.extended_watermark_processor import (
    WatermarkDetector as KGWDetector
)


def get_watermark_detector(base_model: str, **kwargs):
    """Create KGW watermark detector."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

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


class STEAMBODetector:
    """
    Per-Text STEAM BO Detector for optimal pivot language selection.

    For each text:
    1. Sample 3 initial pivots from a genetically diverse pool
    2. Use BO to find optimal pivot language that maximizes normalized z-score
    3. Output: {z_score, prompt, response} where response is text in best pivot language
    """

    def __init__(self,
                 watermark_detector,
                 target_lang: str,
                 input_dir: str,
                 output_dir: str,
                 n_initial: int = 3,
                 max_evaluations: int = 15,
                 random_state: int = 42):
        self.watermark_detector = watermark_detector
        self.target_lang = target_lang
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_initial = n_initial
        self.max_evaluations = max_evaluations
        self.random_state = random_state

        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.backtranslator = RealtimeBacktranslator()
        self.genetic_distance = URIELGeneticDistance()

        # Convert target language to ISO-3 for URIEL
        self.target_lang_iso3 = self._normalize_to_iso3(target_lang)

        # Get available pivot languages (exclude target, filter for Google Translate)
        all_pivots = [lang for lang in self.genetic_distance.available_languages
                     if lang != self.target_lang_iso3]

        from deep_translator import GoogleTranslator
        google_supported = set(GoogleTranslator().get_supported_languages(as_dict=True).values())

        self.available_pivots = []
        for lang in all_pivots:
            try:
                lang_iso1 = iso3_to_iso1(lang)
                if lang_iso1 in google_supported:
                    self.available_pivots.append(lang)
            except ValueError:
                continue

        self.logger.info(f"Filtered to {len(self.available_pivots)} Google Translate supported languages from {len(all_pivots)} total")

        # Cache for validation baselines (pivot_lang -> avg_z_score)
        self._validation_cache = {}

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

    def _sample_initial_pivots(self, text_id: int) -> List[str]:
        """Sample n_initial random pivots from all available languages for a specific text."""
        rng = np.random.RandomState(self.random_state + text_id)
        n_select = min(self.n_initial, len(self.available_pivots))
        return rng.choice(self.available_pivots, n_select, replace=False).tolist()

    def _get_validation_baseline(self, pivot_lang: str) -> float:
        """Get validation baseline z-score for a pivot language from pre-computed files."""
        if pivot_lang in self._validation_cache:
            return self._validation_cache[pivot_lang]

        try:
            pivot_iso1 = iso3_to_iso1(pivot_lang)
        except ValueError:
            self.logger.error(f"Cannot convert {pivot_lang} to ISO-1 for validation file lookup")
            return 0.0

        val_z_score_file = os.path.join(self.input_dir, f"mc4.{pivot_iso1}.val.z_score.jsonl")

        if not os.path.exists(val_z_score_file):
            self.logger.warning(f"Pre-computed validation file not found: {val_z_score_file}")
            return 0.0

        try:
            val_data = read_jsonl(val_z_score_file)
            z_scores = [item.get('z_score', 0.0) for item in val_data if item.get('z_score') is not None]

            if z_scores:
                avg_z_score = sum(z_scores) / len(z_scores)
                self._validation_cache[pivot_lang] = avg_z_score
                self.logger.debug(f"Validation baseline for {pivot_lang} ({pivot_iso1}): {avg_z_score:.4f} "
                                f"(from {len(z_scores)} pre-computed scores)")
                return avg_z_score
            else:
                self.logger.warning(f"No valid z-scores in {val_z_score_file}")
                return 0.0

        except Exception as e:
            self.logger.error(f"Error reading validation file {val_z_score_file}: {e}")
            return 0.0

    def _translate_and_detect(self, text: str, pivot_lang: str) -> Tuple[float, str, bool]:
        """
        Translate text to pivot language and detect watermark.

        Returns:
            Tuple of (raw_z_score, translated_text, success)
        """
        try:
            pivot_iso1 = iso3_to_iso1(pivot_lang)
            target_iso1 = iso3_to_iso1(self.target_lang_iso3)

            if not pivot_iso1 or not target_iso1:
                self.logger.error(f"Language code conversion failed: {pivot_lang} or {self.target_lang_iso3}")
                return 0.0, "", False

            # Translate: target_lang → pivot_lang
            translated_text = self.backtranslator.translate_text(text, target_iso1, pivot_iso1)

            if translated_text is None:
                self.logger.error(f"Translation failed: {target_iso1} → {pivot_iso1}")
                return 0.0, "", False

            # Detect watermark
            detection_result = self.watermark_detector.detect(translated_text)
            raw_z_score = detection_result.get('z_score', 0.0)

            return raw_z_score, translated_text, True

        except Exception as e:
            self.logger.error(f"Error in translate_and_detect for {pivot_lang}: {e}")
            return 0.0, "", False

    def _evaluate_pivot_language(self, text: str, pivot_lang: str) -> Dict[str, Any]:
        """Evaluate a pivot language for a specific text."""
        raw_z_score, translated_text, success = self._translate_and_detect(text, pivot_lang)

        if not success:
            return {
                'pivot_lang': pivot_lang,
                'raw_z_score': 0.0,
                'normalized_z_score': 0.0,
                'genetic_distance': 0.0,
                'translated_text': '',
                'success': False
            }

        # Get validation baseline and normalize
        validation_baseline = self._get_validation_baseline(pivot_lang)
        normalized_z_score = raw_z_score - validation_baseline

        # Get genetic distance
        try:
            genetic_distance = self.genetic_distance.get_genetic_distance(
                self.target_lang_iso3, pivot_lang
            )
        except Exception as e:
            self.logger.warning(f"Error getting genetic distance for {pivot_lang}: {e}")
            genetic_distance = 3.0

        return {
            'pivot_lang': pivot_lang,
            'raw_z_score': raw_z_score,
            'normalized_z_score': normalized_z_score,
            'genetic_distance': genetic_distance,
            'translated_text': translated_text,
            'success': True
        }

    def _bo_suggest_next_pivot(self, evaluations: List[Dict[str, Any]]) -> str:
        """Use BO to suggest next best pivot language."""
        if len(evaluations) < 2:
            evaluated_pivots = {e['pivot_lang'] for e in evaluations}
            remaining = [lang for lang in self.available_pivots if lang not in evaluated_pivots]
            if remaining:
                return np.random.choice(remaining)
            return self.available_pivots[0]

        try:
            X_samples = []
            y_samples = []

            for e in evaluations:
                if e['success']:
                    X_samples.append([e['genetic_distance']])
                    y_samples.append(e['normalized_z_score'])

            if len(X_samples) < 2:
                evaluated_pivots = {e['pivot_lang'] for e in evaluations}
                remaining = [lang for lang in self.available_pivots if lang not in evaluated_pivots]
                return np.random.choice(remaining) if remaining else self.available_pivots[0]

            search_space = [Real(0.0, 6.0, name='genetic_distance')]

            # gp_minimize fits a GP on (x0, y0) and uses EI to suggest next point
            # We negate y because gp_minimize minimizes, but we want to maximize normalized_z_score
            result = gp_minimize(
                func=lambda x: 0.0,  # Dummy — not called with n_calls=1, n_initial_points=0
                dimensions=search_space,
                n_calls=1,
                n_initial_points=0,
                x0=X_samples,
                y0=[-y for y in y_samples],
                acq_func='EI',
                random_state=self.random_state
            )

            suggested_distance = result.x[0]
            return self._find_pivot_at_distance(suggested_distance, evaluations)

        except Exception as e:
            self.logger.error(f"BO suggestion failed: {e}")
            evaluated_pivots = {e_['pivot_lang'] for e_ in evaluations}
            remaining = [lang for lang in self.available_pivots if lang not in evaluated_pivots]
            if remaining:
                return np.random.choice(remaining)
            return self.available_pivots[0]

    def _find_pivot_at_distance(self, target_distance: float, evaluations: List[Dict[str, Any]]) -> str:
        """Find pivot language closest to target genetic distance."""
        evaluated_pivots = {e['pivot_lang'] for e in evaluations}

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

    def optimize_single_text(self, text: str, prompt: str, text_id: int) -> Dict[str, Any]:
        """
        Run STEAM BO optimization for a single text.

        Returns:
            Dict with {z_score, prompt, response} for the best pivot language
        """
        self.logger.info(f"Starting STEAM BO for text {text_id}")

        evaluations = []

        # Phase 1: Evaluate initial pivot languages (sampled from diverse pool)
        initial_pivots = self._sample_initial_pivots(text_id)
        self.logger.info(f"  Initial pivots for text {text_id}: {initial_pivots}")

        for pivot_lang in initial_pivots:
            eval_result = self._evaluate_pivot_language(text, pivot_lang)
            evaluations.append(eval_result)
            self.logger.info(f"  Initial {pivot_lang}: norm_z={eval_result['normalized_z_score']:.3f}, "
                           f"dist={eval_result['genetic_distance']:.3f}")

        # Find current best
        successful_evals = [e for e in evaluations if e['success']]
        if not successful_evals:
            self.logger.error(f"No successful evaluations for text {text_id}")
            return {'z_score': 0.0, 'prompt': prompt, 'response': text}

        best_eval = max(successful_evals, key=lambda x: x['normalized_z_score'])

        # Phase 2: BO optimization loop
        for iteration in range(self.max_evaluations - self.n_initial):
            next_pivot = self._bo_suggest_next_pivot(evaluations)
            if not next_pivot:
                break

            self.logger.info(f"  BO iteration {iteration + 1}: trying {next_pivot}")

            eval_result = self._evaluate_pivot_language(text, next_pivot)
            evaluations.append(eval_result)

            if eval_result['success'] and eval_result['normalized_z_score'] > best_eval['normalized_z_score']:
                best_eval = eval_result

            self.logger.info(f"    {next_pivot}: norm_z={eval_result['normalized_z_score']:.3f}, "
                           f"dist={eval_result['genetic_distance']:.3f}")

        self.logger.info(f"  Text {text_id}: best pivot={best_eval['pivot_lang']}, "
                        f"norm_z={best_eval['normalized_z_score']:.3f}")

        return {
            'z_score': best_eval['normalized_z_score'],
            'prompt': prompt,
            'response': best_eval['translated_text']
        }

    def run(self, num_texts: int = 500) -> str:
        """
        Run STEAM BO optimization for all texts and write output JSONL.

        Args:
            num_texts: Number of texts to process

        Returns:
            Path to the output JSONL file
        """
        self.logger.info(f"Starting STEAM BO optimization for {num_texts} texts")

        # Load watermarked texts
        mod_file = os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.mod.jsonl")
        if not os.path.exists(mod_file):
            raise FileNotFoundError(f"Input file not found: {mod_file}")

        mod_data = read_jsonl(mod_file)[:num_texts]

        # Output file
        output_file = os.path.join(self.output_dir, f"mc4.{self.target_lang}.bo.z_score.jsonl")

        with open(output_file, 'w') as f:
            for i, item in enumerate(mod_data):
                text_content = item.get('response', '')
                prompt = item.get('prompt', '')

                if not text_content:
                    self.logger.warning(f"Empty text at index {i}, skipping")
                    continue

                try:
                    result = self.optimize_single_text(text_content, prompt, i)
                    f.write(json.dumps(result) + '\n')
                except Exception as e:
                    self.logger.error(f"Error processing text {i}: {e}")
                    # Write fallback entry so indices stay aligned
                    f.write(json.dumps({'z_score': 0.0, 'prompt': prompt, 'response': text_content}) + '\n')

        self.logger.info(f"Results saved to {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description="STEAM BO Detector")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n_initial", type=int, default=3, help="Initial pivot languages")
    parser.add_argument("--max_evaluations", type=int, default=15, help="Max BO evaluations")
    parser.add_argument("--num_texts", type=int, default=500, help="Number of texts to process")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    watermark_detector = get_watermark_detector(base_model=args.base_model)

    # Initialize and run STEAM BO detector
    steam_detector = STEAMBODetector(
        watermark_detector=watermark_detector,
        target_lang=args.tgt_lang,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_initial=args.n_initial,
        max_evaluations=args.max_evaluations,
        random_state=args.random_state
    )

    output_file = steam_detector.run(num_texts=args.num_texts)
    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    main()
