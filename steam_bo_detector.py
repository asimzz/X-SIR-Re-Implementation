#!/usr/bin/env python3
"""
STEAM BO Detector - Per-Text Bayesian Optimization for Pivot Language Selection

This implements the correct STEAM BO approach:
1. For each individual text, run separate BO optimization
2. Find optimal pivot language that maximizes γ_lang-corrected z-score
3. Translation flow: tgt_lang → pivot_lang (single step)
4. Z-score correction: use per-language γ_lang in the z-score formula
   (no post-hoc normalization needed)

Output:
  - mc4.{target_lang}.bo.z_score.jsonl      (watermarked texts, BO-optimized)
  - mc4.{target_lang}.bo.hum.z_score.jsonl  (human texts, matched pivot from BO)

Author: Asim
"""

import os
import json
import math
import numpy as np
import logging
import argparse
from typing import Dict, List, Any, Tuple

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

# Import your existing components
from realtime_backtranslation import RealtimeBacktranslator
from language_features import LanguageFeatures
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
        gamma=kwargs.get('gamma', 0.25),
        seed=kwargs.get('seed', 0),
        seeding_scheme=kwargs.get('seeding_scheme', 'minhash'),
        device=device,
        tokenizer=tokenizer,
        z_threshold=kwargs.get('z_threshold', 4.0),
        normalizers=kwargs.get('normalizers', []),
        ignore_repeated_ngrams=kwargs.get('ignore_repeated_ngrams', True),
    )


def _count_complete_lines_and_truncate(path: str) -> int:
    """Return the number of complete (newline-terminated) lines in a JSONL file.

    If a trailing partial line is present (interrupted mid-write), the file is
    rewritten to drop it so subsequent appends produce valid JSONL.
    """
    if not os.path.exists(path):
        return 0
    with open(path, 'rb') as f:
        data = f.read()
    if not data:
        return 0
    parts = data.split(b'\n')
    # If file ends in '\n', the final element is b'' → no partial line.
    complete = parts[:-1]
    if parts[-1] != b'':
        with open(path, 'wb') as f:
            f.write(b'\n'.join(complete) + (b'\n' if complete else b''))
    return len(complete)


def _truncate_to_n_lines(path: str, n: int) -> None:
    """Rewrite `path` to contain only its first `n` complete lines."""
    if not os.path.exists(path):
        return
    with open(path, 'rb') as f:
        data = f.read()
    parts = data.split(b'\n')
    keep = parts[:n]
    with open(path, 'wb') as f:
        f.write(b'\n'.join(keep) + (b'\n' if keep else b''))


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
                 gamma_lang_file: str,
                 n_initial: int = 3,
                 max_evaluations: int = 15,
                 random_state: int = 42,
                 per_example_attack: bool = False,
                 mod_file: str = None,
                 hum_file: str = None):
        self.watermark_detector = watermark_detector
        self.target_lang = target_lang
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_initial = n_initial
        self.max_evaluations = max_evaluations
        self.random_state = random_state
        # Per-example-attack mode: each input line carries its own `attack_lang`
        # (the language it was translated into). The source language S is passed
        # as `target_lang` and is kept as an eligible pivot (back-translating
        # attack_lang -> S recovers the watermark's native token space).
        self.per_example_attack = per_example_attack
        self.mod_file = mod_file
        self.hum_file = hum_file

        # Setup logging first so all messages are visible
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load γ_lang values (language-specific green token fractions)
        print("Loading gamma_lang file...")
        with open(gamma_lang_file, 'r') as f:
            self._gamma_lang_data = json.load(f)
        self.logger.info(f"Loaded γ_lang for {len(self._gamma_lang_data)} languages from {gamma_lang_file}")

        # Initialize components
        print("Initializing backtranslator...")
        self.backtranslator = RealtimeBacktranslator()
        print("Initializing language features...")
        self.lang_features = LanguageFeatures(feature_sets=['syntax_knn', 'phonology_knn'])
        print("Language features loaded.")

        # Convert target language to ISO-3 for URIEL
        self.target_lang_iso3 = self._normalize_to_iso3(target_lang)

        # Load supported languages (ISO-1 codes from supported_languages.txt)
        with open('supported_languages.txt', 'r') as f:
            supported_iso1 = [line.strip() for line in f if line.strip()]

        # Build available pivots: convert to ISO-3, exclude target, filter for URIEL
        self.available_pivots = []
        self.available_pivots_iso1 = []
        for lang_iso1 in supported_iso1:
            try:
                lang_iso3 = iso1_to_iso3(lang_iso1)
            except ValueError:
                continue
            # In per-example-attack mode the source language S (passed as
            # target_lang) must stay in the pivot pool — it is typically the
            # best pivot. Otherwise keep the original behaviour of excluding it.
            if not self.per_example_attack and lang_iso3 == self.target_lang_iso3:
                continue
            if lang_iso3 in self.lang_features.available_languages:
                self.available_pivots.append(lang_iso3)
                self.available_pivots_iso1.append(lang_iso1)

        self.logger.info(f"Available pivot languages: {len(self.available_pivots)} (from {len(supported_iso1)} supported)")

        # Pre-compute feature vectors for all pivot languages
        print(f"Pre-computing feature vectors for {len(self.available_pivots)} pivots...")
        self._feature_vectors = {}
        for lang in self.available_pivots:
            self._feature_vectors[lang] = self.lang_features.get_feature_vector(lang)

        self.feature_dim = len(next(iter(self._feature_vectors.values())))
        print(f"Feature vectors ready: {self.feature_dim} dimensions")

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

    def _get_gamma_lang(self, pivot_lang: str) -> float:
        """
        Get γ_lang for a pivot language.

        Returns the empirical green token fraction from calibration data.
        Falls back to the detector's default gamma (0.25) if not available.
        """
        try:
            pivot_iso1 = iso3_to_iso1(pivot_lang)
        except ValueError:
            self.logger.warning(f"Cannot convert {pivot_lang} to ISO-1, using default gamma")
            return self.watermark_detector.gamma

        if pivot_iso1 in self._gamma_lang_data:
            return self._gamma_lang_data[pivot_iso1]["gamma_lang"]

        self.logger.warning(f"No γ_lang for {pivot_iso1}, using default gamma={self.watermark_detector.gamma}")
        return self.watermark_detector.gamma

    def _recompute_z_score(self, num_green_tokens: int, num_tokens_scored: int, gamma_lang: float) -> float:
        """Recompute z-score using language-specific γ_lang."""
        numer = num_green_tokens - gamma_lang * num_tokens_scored
        denom = math.sqrt(num_tokens_scored * gamma_lang * (1 - gamma_lang))
        return numer / denom

    def _translate_and_detect(self, text: str, pivot_lang: str,
                              src_lang_iso1: str = None) -> Tuple[float, str, bool]:
        """
        Translate text (in `src_lang_iso1`) to pivot language and detect watermark.

        `src_lang_iso1` is the language the text is currently in — the attack
        language for a given example. When None it defaults to the detector's
        target language (the original English-source behaviour).

        The z-score is recomputed using γ_lang (language-specific green token fraction)
        instead of the default γ=0.25, which corrects for tokenizer bias.

        Returns:
            Tuple of (corrected_z_score, translated_text, success)
        """
        try:
            pivot_iso1 = iso3_to_iso1(pivot_lang)
            if src_lang_iso1 is None:
                src_lang_iso1 = iso3_to_iso1(self.target_lang_iso3)

            if not pivot_iso1 or not src_lang_iso1:
                self.logger.error(f"Language code conversion failed: {pivot_lang} or {src_lang_iso1}")
                return 0.0, "", False

            # Identity pivot: text is already in the pivot language, so there is
            # no recovery translation to do — score the text as-is (the cheap
            # "no back-translation" candidate). Normalise both sides so that
            # e.g. zh/zh-CN and iw/he compare equal.
            if self.backtranslator.normalize_lang_code(pivot_iso1) == \
                    self.backtranslator.normalize_lang_code(src_lang_iso1):
                translated_text = text
            else:
                # Translate: src_lang → pivot_lang
                translated_text = self.backtranslator.translate_text(text, src_lang_iso1, pivot_iso1)

            if translated_text is None:
                self.logger.error(f"Translation failed: {src_lang_iso1} → {pivot_iso1}")
                return 0.0, "", False

            # Detect watermark (uses default γ=0.25 for green list partitioning)
            detection_result = self.watermark_detector.detect(translated_text)
            num_green = detection_result.get('num_green_tokens')
            num_scored = detection_result.get('num_tokens_scored')

            if num_green is None or num_scored is None or num_scored == 0:
                return 0.0, "", False

            # Recompute z-score using γ_lang
            gamma_lang = self._get_gamma_lang(pivot_lang)
            corrected_z = self._recompute_z_score(int(num_green), int(num_scored), gamma_lang)

            return corrected_z, translated_text, True

        except Exception as e:
            self.logger.error(f"Error in translate_and_detect for {pivot_lang}: {e}")
            return 0.0, "", False

    def _evaluate_pivot_language(self, text: str, pivot_lang: str,
                                 src_lang_iso1: str = None) -> Dict[str, Any]:
        """Evaluate a pivot language for a specific text."""
        z_score, translated_text, success = self._translate_and_detect(text, pivot_lang, src_lang_iso1)

        if not success:
            return {
                'pivot_lang': pivot_lang,
                'z_score': 0.0,
                'feature_vector': self._feature_vectors.get(pivot_lang, np.zeros(self.feature_dim)),
                'translated_text': '',
                'success': False
            }

        return {
            'pivot_lang': pivot_lang,
            'z_score': z_score,
            'feature_vector': self._feature_vectors[pivot_lang],
            'translated_text': translated_text,
            'success': True
        }

    def _bo_suggest_next_pivot(self, evaluations: List[Dict[str, Any]]) -> str:
        """
        Use BoTorch GP + LogEI to suggest next pivot language.

        Instead of optimizing the acquisition function in continuous space and mapping
        to the nearest language (which breaks in 131-D with ~100 discrete candidates),
        we evaluate the acquisition function directly at all unevaluated language
        feature vectors and pick the one with highest Expected Improvement.
        """
        evaluated_pivots = {e['pivot_lang'] for e in evaluations}
        remaining = [lang for lang in self.available_pivots if lang not in evaluated_pivots]

        if not remaining:
            return None

        # Need at least 2 successful evaluations to fit a GP
        successful = [e for e in evaluations if e['success']]
        if len(successful) < 2:
            return np.random.choice(remaining)

        try:
            X_samples = [list(e['feature_vector']) for e in successful]
            y_samples = [e['z_score'] for e in successful]

            train_X = torch.from_numpy(np.array(X_samples)).double()
            train_Y = torch.tensor(y_samples, dtype=torch.double).unsqueeze(-1)

            # Fit GP surrogate model
            gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Build tensor of ALL unevaluated candidate feature vectors
            candidate_features = torch.tensor(
                [self._feature_vectors[lang].tolist() for lang in remaining],
                dtype=torch.double
            )

            # Evaluate acquisition function at each discrete candidate
            best_f = train_Y.max()
            ei = LogExpectedImprovement(gp, best_f=best_f)
            # LogEI expects shape (batch, q, d) — add q=1 dimension
            ei_values = ei(candidate_features.unsqueeze(1))
            best_idx = ei_values.argmax().item()

            return remaining[best_idx]

        except Exception as e:
            self.logger.error(f"BO suggestion failed: {e}")
            return np.random.choice(remaining)

    def optimize_single_text(self, text: str, prompt: str, text_id: int,
                             src_lang_iso1: str = None) -> Dict[str, Any]:
        """
        Run STEAM BO optimization for a single text.

        `src_lang_iso1` is the language the text is currently in (the example's
        attack language). When None it defaults to the detector target language.

        Returns:
            Dict with {z_score, prompt, response} for the best pivot language
        """
        self.logger.info(f"Starting STEAM BO for text {text_id} (source lang={src_lang_iso1 or self.target_lang})")

        evaluations = []

        # Phase 1: Evaluate initial pivot languages (sampled from diverse pool)
        initial_pivots = self._sample_initial_pivots(text_id)
        self.logger.info(f"  Initial pivots for text {text_id}: {initial_pivots}")

        for pivot_lang in initial_pivots:
            eval_result = self._evaluate_pivot_language(text, pivot_lang, src_lang_iso1)
            evaluations.append(eval_result)
            self.logger.info(f"  Initial {pivot_lang}: z={eval_result['z_score']:.3f}")

        # Find current best (by γ_lang-corrected z-score)
        successful_evals = [e for e in evaluations if e['success']]
        if not successful_evals:
            self.logger.error(f"No successful evaluations for text {text_id}")
            return {'z_score': 0.0, 'prompt': prompt, 'response': text}

        best_eval = max(successful_evals, key=lambda x: x['z_score'])

        # Phase 2: BO optimization loop
        for iteration in range(self.max_evaluations - self.n_initial):
            next_pivot = self._bo_suggest_next_pivot(evaluations)
            if not next_pivot:
                break

            self.logger.info(f"  BO iteration {iteration + 1}: trying {next_pivot}")

            eval_result = self._evaluate_pivot_language(text, next_pivot, src_lang_iso1)
            evaluations.append(eval_result)

            if eval_result['success'] and eval_result['z_score'] > best_eval['z_score']:
                best_eval = eval_result

            self.logger.info(f"    {next_pivot}: z={eval_result['z_score']:.3f}")

        self.logger.info(f"  Text {text_id}: best pivot={best_eval['pivot_lang']}, z={best_eval['z_score']:.3f}")

        # Convert best pivot to ISO-1 for readability in output
        best_pivot_iso3 = best_eval['pivot_lang']
        try:
            best_pivot_iso1 = iso3_to_iso1(best_pivot_iso3)
        except ValueError:
            best_pivot_iso1 = best_pivot_iso3

        return {
            'z_score': best_eval['z_score'],
            'best_pivot': best_pivot_iso1,
            'prompt': prompt,
            'response': best_eval['translated_text']
        }

    def _process_human_text(self, text: str, prompt: str, best_pivot_iso3: str,
                            src_lang_iso1: str = None) -> Dict[str, Any]:
        """Translate human text to the BO-selected pivot and detect with γ_lang correction."""
        z_score, translated_text, success = self._translate_and_detect(text, best_pivot_iso3, src_lang_iso1)

        if not success:
            return {'z_score': 0.0, 'prompt': prompt, 'response': text}

        return {
            'z_score': z_score,
            'prompt': prompt,
            'response': translated_text
        }

    def run(self, num_texts: int = 500) -> str:
        """
        Run STEAM BO on watermarked texts, then apply the selected pivot
        to corresponding human texts.

        Produces:
          - mc4.{target_lang}.bo.z_score.jsonl      (watermarked, BO-optimized)
          - mc4.{target_lang}.bo.hum.z_score.jsonl   (human, matched pivot)
        """
        self.logger.info(f"Starting STEAM BO for {num_texts} texts")

        # Resolve input files. When explicit --mod_file/--hum_file are given
        # (per-example-attack mode / arbitrary inputs) use them directly;
        # otherwise fall back to the English-source naming convention.
        if self.mod_file:
            mod_file = self.mod_file
        else:
            mod_file = os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.mod.jsonl")
        if not os.path.exists(mod_file):
            raise FileNotFoundError(f"Input file not found: {mod_file}")
        mod_data = read_jsonl(mod_file)[:num_texts]

        if self.hum_file:
            hum_file = self.hum_file
        else:
            hum_file = os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.hum.jsonl")
        if not os.path.exists(hum_file):
            raise FileNotFoundError(f"Human text file not found: {hum_file}")
        hum_data = read_jsonl(hum_file)[:num_texts]

        # In per-example mode --tgt_lang is the SOURCE language S; tag outputs "-mix".
        out_stub = f"{self.target_lang}-mix" if self.per_example_attack else self.target_lang
        mod_output = os.path.join(self.output_dir, f"mc4.{out_stub}.bo.z_score.jsonl")
        hum_output = os.path.join(self.output_dir, f"mc4.{out_stub}.bo.hum.z_score.jsonl")

        # Default translation source (English-source mode): the target language.
        default_src_iso1 = iso3_to_iso1(self.target_lang_iso3)

        # Resume: drop any partial trailing lines, then realign the two files
        # to a common index so appends stay in lockstep.
        mod_count = _count_complete_lines_and_truncate(mod_output)
        hum_count = _count_complete_lines_and_truncate(hum_output)
        start_idx = min(mod_count, hum_count)
        if mod_count != hum_count:
            self.logger.warning(
                f"Output files out of sync (mod={mod_count}, hum={hum_count}); "
                f"truncating both to {start_idx} to realign."
            )
            if mod_count > start_idx:
                _truncate_to_n_lines(mod_output, start_idx)
            if hum_count > start_idx:
                _truncate_to_n_lines(hum_output, start_idx)

        if start_idx >= num_texts:
            self.logger.info(
                f"All {num_texts} texts already processed for {self.target_lang}, skipping"
            )
            return mod_output

        if start_idx > 0:
            self.logger.info(f"Resuming from text {start_idx} ({start_idx} already processed)")

        with open(mod_output, 'a') as f_mod, open(hum_output, 'a') as f_hum:
            for i, item in enumerate(mod_data):
                if i < start_idx:
                    continue
                text_content = item.get('response', '')
                prompt = item.get('prompt', '')

                if not text_content:
                    self.logger.warning(f"Empty text at index {i}, skipping")
                    continue

                # Per-example source (attack) language; fall back to the target
                # language for the legacy English-source layout.
                mod_src_iso1 = item.get('attack_lang') or default_src_iso1
                mod_tier = item.get('attack_tier')

                try:
                    # BO on watermarked text
                    result = self.optimize_single_text(text_content, prompt, i, src_lang_iso1=mod_src_iso1)
                    if item.get('attack_lang'):
                        result['attack_lang'] = item['attack_lang']
                        result['attack_tier'] = mod_tier
                    f_mod.write(json.dumps(result) + '\n')

                    # Apply same pivot to human text (no BO, just translate + detect)
                    best_pivot_iso1 = result['best_pivot']
                    best_pivot_iso3 = iso1_to_iso3(best_pivot_iso1)
                    if i < len(hum_data):
                        hum_text = hum_data[i].get('response', '')
                        hum_prompt = hum_data[i].get('prompt', '')
                        hum_src_iso1 = hum_data[i].get('attack_lang') or default_src_iso1
                        hum_result = self._process_human_text(
                            hum_text, hum_prompt, best_pivot_iso3, src_lang_iso1=hum_src_iso1)
                        if hum_data[i].get('attack_lang'):
                            hum_result['attack_lang'] = hum_data[i]['attack_lang']
                            hum_result['attack_tier'] = hum_data[i].get('attack_tier')
                    else:
                        hum_result = {'z_score': 0.0, 'prompt': '', 'response': ''}
                    f_hum.write(json.dumps(hum_result) + '\n')

                except Exception as e:
                    self.logger.error(f"Error processing text {i}: {e}")
                    err_mod = {'z_score': 0.0, 'prompt': prompt, 'response': text_content}
                    err_hum = {'z_score': 0.0, 'prompt': '', 'response': ''}
                    if item.get('attack_lang'):
                        err_mod['attack_lang'] = item['attack_lang']
                        err_mod['attack_tier'] = mod_tier
                        err_hum['attack_lang'] = item['attack_lang']
                        err_hum['attack_tier'] = mod_tier
                    f_mod.write(json.dumps(err_mod) + '\n')
                    f_hum.write(json.dumps(err_hum) + '\n')

        self.logger.info(f"Watermarked results: {mod_output}")
        self.logger.info(f"Human results:       {hum_output}")
        return mod_output


def main():
    parser = argparse.ArgumentParser(description="STEAM BO Detector")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gamma_lang_file", type=str, required=True, help="Path to gamma_lang.json")
    parser.add_argument("--n_initial", type=int, default=3, help="Initial pivot languages")
    parser.add_argument("--max_evaluations", type=int, default=15, help="Max BO evaluations")
    parser.add_argument("--num_texts", type=int, default=500, help="Number of texts to process")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--per_example_attack", action="store_true",
                        help="Mixed-attack mode: each input line has its own 'attack_lang'; "
                             "--tgt_lang is the SOURCE language S (kept as an eligible pivot); "
                             "outputs are named mc4.{S}-mix.bo(.hum).z_score.jsonl")
    parser.add_argument("--mod_file", type=str, default=None,
                        help="Explicit watermarked input file (overrides mc4.en-{tgt}.mod.jsonl)")
    parser.add_argument("--hum_file", type=str, default=None,
                        help="Explicit human input file (overrides mc4.en-{tgt}.hum.jsonl)")

    args = parser.parse_args()

    watermark_detector = get_watermark_detector(base_model=args.base_model)

    steam_detector = STEAMBODetector(
        watermark_detector=watermark_detector,
        target_lang=args.tgt_lang,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gamma_lang_file=args.gamma_lang_file,
        n_initial=args.n_initial,
        max_evaluations=args.max_evaluations,
        random_state=args.random_state,
        per_example_attack=args.per_example_attack,
        mod_file=args.mod_file,
        hum_file=args.hum_file,
    )

    output_file = steam_detector.run(num_texts=args.num_texts)
    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    main()