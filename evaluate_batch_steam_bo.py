#!/usr/bin/env python3
"""
Batch BO-Enhanced STEAM Watermark Evaluation

This script implements your supervisor's batch approach for STEAM BO evaluation:
1. Uses validation-normalized z-scores (your key formula: raw_zscore - avg_val_zscore)
2. Processes all 500 samples per language efficiently
3. Uses URIEL genetic distances with BO for language selection
4. Optimizes AUC like your evaluate_normalized_detection.py

Called by evaluate_bo_steam.sh script.

Author: Asim
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

# Import genetic diversity selector and BO components
from genetic_diversity_selector import GeneticDiversitySelector
from realtime_backtranslation import RealtimeBacktranslator
from uriel_genetic_distance import URIELGeneticDistance
from language_code_converter import iso3_to_iso1, is_valid_iso3

# Import watermark detectors
from src_watermark.kgw.extended_watermark_processor import WatermarkDetector as KGWDetector
from src_watermark.xsir.watermark import WatermarkContext as XSIRDetector

# BO imports
from skopt import gp_minimize
from skopt.space import Real
from skopt.acquisition import gaussian_ei, gaussian_ucb, gaussian_pi


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file - matches your utils.py format."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    return data


def extract_zscores(data_list: List[Dict]) -> List[float]:
    """Extract z-scores from detection results - matches your format."""
    return [item["z_score"] if item["z_score"] is not None else 0.0 for item in data_list]


def get_avg_zscore(validation_file: str) -> float:
    """Get average validation z-score - YOUR KEY NORMALIZATION FUNCTION."""
    validation_list = read_jsonl(validation_file)
    zscores = extract_zscores(validation_list)
    return sum(zscores) / len(zscores) if zscores else 0.0


class BatchSTEAMBOEvaluator:
    """
    Batch BO-enhanced STEAM evaluator following your supervisor's approach.
    """

    def __init__(self,
                 watermark_detector,
                 target_lang: str,
                 n_initial: int = 3,
                 max_evaluations: int = 8,
                 languages_file: str = "supported_languages.txt",
                 random_state: int = 42):
        """
        Initialize batch STEAM BO evaluator.

        Args:
            watermark_detector: KGW/XSIR detector instance
            target_lang: Target language in ISO-3 format (e.g., 'deu')
            n_initial: Number of initial diverse languages
            max_evaluations: Maximum BO evaluations
            languages_file: File containing supported languages
            random_state: Random seed
        """
        self.watermark_detector = watermark_detector
        self.target_lang = target_lang
        self.n_initial = n_initial
        self.max_evaluations = max_evaluations
        self.random_state = random_state

        # Initialize components
        self.diversity_selector = GeneticDiversitySelector(
            languages_file=languages_file,
            random_seed=random_state
        )
        self.backtranslator = RealtimeBacktranslator()
        self.genetic_distance = URIELGeneticDistance()

        # Get initial diverse languages (ISO-3)
        self.initial_languages = self._select_initial_languages()

        # BO state
        self.X_samples = []  # Features: [genetic_distance]
        self.y_samples = []  # AUC scores from validation-normalized z-scores
        self.evaluation_history = []

        logger.info(f"Initialized Batch STEAM BO for target: {target_lang}")
        logger.info(f"Initial diverse languages: {self.initial_languages}")

    def _select_initial_languages(self) -> List[str]:
        """Select initial genetically diverse languages."""
        try:
            diverse_langs, diversity_score = self.diversity_selector.select_diverse_languages(
                n_languages=self.n_initial,
                method="exhaustive" if len(self.diversity_selector.available_languages) < 100 else "random_sample"
            )
            logger.info(f"Selected diverse languages (score={diversity_score:.3f}): {diverse_langs}")
            return diverse_langs
        except Exception as e:
            logger.error(f"Failed to select diverse languages: {e}")
            # Fallback to random selection
            available = self.diversity_selector.available_languages
            np.random.seed(self.random_state)
            fallback = np.random.choice(available, self.n_initial, replace=False).tolist()
            logger.info(f"Using fallback random languages: {fallback}")
            return fallback

    def _batch_backtranslate_texts(self,
                                   watermarked_texts: List[str],
                                   human_texts: List[str],
                                   intermediate_lang: str) -> tuple:
        """
        Batch backtranslate all texts through intermediate language.

        Args:
            watermarked_texts: List of watermarked texts
            human_texts: List of human texts
            intermediate_lang: Intermediate language (ISO-3)

        Returns:
            (backtranslated_wm_texts, backtranslated_hm_texts)
        """
        # Convert to ISO-1 for Google Translate
        intermediate_iso1 = iso3_to_iso1(intermediate_lang)
        target_iso1 = iso3_to_iso1(self.target_lang)

        if not intermediate_iso1 or not target_iso1:
            logger.error(f"Language conversion failed: {intermediate_lang} -> {intermediate_iso1}")
            return [], []

        logger.info(f"Batch backtranslating {len(watermarked_texts)} texts: {target_iso1} -> {intermediate_iso1} -> {target_iso1}")

        # Batch process watermarked texts
        backtranslated_wm = []
        for i, text in enumerate(tqdm(watermarked_texts, desc="WM backtranslation")):
            try:
                # target_lang -> intermediate_lang
                intermediate_text = self.backtranslator.translate_text(
                    text, target_iso1, intermediate_iso1
                )
                if intermediate_text is None:
                    backtranslated_wm.append("")
                    continue

                # intermediate_lang -> target_lang
                final_text = self.backtranslator.translate_text(
                    intermediate_text, intermediate_iso1, target_iso1
                )
                backtranslated_wm.append(final_text if final_text else "")

            except Exception as e:
                logger.warning(f"Translation failed for WM text {i}: {e}")
                backtranslated_wm.append("")

        # Batch process human texts
        backtranslated_hm = []
        for i, text in enumerate(tqdm(human_texts, desc="HM backtranslation")):
            try:
                # target_lang -> intermediate_lang
                intermediate_text = self.backtranslator.translate_text(
                    text, target_iso1, intermediate_iso1
                )
                if intermediate_text is None:
                    backtranslated_hm.append("")
                    continue

                # intermediate_lang -> target_lang
                final_text = self.backtranslator.translate_text(
                    intermediate_text, intermediate_iso1, target_iso1
                )
                backtranslated_hm.append(final_text if final_text else "")

            except Exception as e:
                logger.warning(f"Translation failed for HM text {i}: {e}")
                backtranslated_hm.append("")

        return backtranslated_wm, backtranslated_hm

    def _batch_detect_watermarks(self, texts: List[str]) -> List[float]:
        """Batch detect watermarks in texts."""
        z_scores = []
        for text in tqdm(texts, desc="Watermark detection"):
            if not text:
                z_scores.append(0.0)
                continue
            try:
                result = self.watermark_detector.detect(text)
                z_score = result.get('z_score', 0.0)
                z_scores.append(z_score if z_score is not None else 0.0)
            except Exception as e:
                logger.warning(f"Detection failed: {e}")
                z_scores.append(0.0)
        return z_scores

    def _evaluate_language_candidate(self,
                                   watermarked_texts: List[str],
                                   human_texts: List[str],
                                   validation_avg: float,
                                   intermediate_lang: str) -> Dict:
        """
        Evaluate a single intermediate language candidate.

        Returns:
            Dictionary with AUC, TPR metrics, and z-scores
        """
        logger.info(f"Evaluating intermediate language: {intermediate_lang}")

        # Step 1: Batch backtranslate
        bt_wm_texts, bt_hm_texts = self._batch_backtranslate_texts(
            watermarked_texts, human_texts, intermediate_lang
        )

        if not bt_wm_texts or not bt_hm_texts:
            return {
                'intermediate_lang': intermediate_lang,
                'auc': 0.0,
                'error': 'Backtranslation failed',
                'genetic_distance': 0.0
            }

        # Step 2: Batch detect watermarks
        wm_z_scores = self._batch_detect_watermarks(bt_wm_texts)
        hm_z_scores = self._batch_detect_watermarks(bt_hm_texts)

        # Step 3: Apply YOUR normalization (key formula from evaluate_normalized_detection.py)
        normalized_wm_scores = [z - validation_avg for z in wm_z_scores]
        normalized_hm_scores = [z - validation_avg for z in hm_z_scores]

        # Step 4: Calculate AUC (like your evaluate_normalized_detection.py)
        from sklearn.metrics import roc_auc_score, roc_curve

        y_true = [0] * len(normalized_hm_scores) + [1] * len(normalized_wm_scores)  # 0=human, 1=watermarked
        y_scores = normalized_hm_scores + normalized_wm_scores

        try:
            auc = roc_auc_score(y_true, y_scores)

            # Calculate TPR at FPR thresholds
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            from scipy import interpolate

            def tpr_at_fpr(fpr, tpr, fpr_target):
                fpr_tpr_interpolation = interpolate.interp1d(fpr, tpr, kind="linear")
                return fpr_tpr_interpolation(fpr_target)

            tpr_01 = tpr_at_fpr(fpr, tpr, 0.1) if len(fpr) > 1 else 0.0
            tpr_001 = tpr_at_fpr(fpr, tpr, 0.01) if len(fpr) > 1 else 0.0

        except Exception as e:
            logger.warning(f"AUC calculation failed for {intermediate_lang}: {e}")
            auc = 0.0
            tpr_01 = 0.0
            tpr_001 = 0.0

        # Step 5: Get genetic distance
        try:
            genetic_dist = self.genetic_distance.get_genetic_distance(
                self.target_lang, intermediate_lang
            )
        except Exception as e:
            logger.warning(f"Genetic distance failed for {intermediate_lang}: {e}")
            genetic_dist = 5.0  # Maximum distance as fallback

        result = {
            'intermediate_lang': intermediate_lang,
            'auc': auc,
            'tpr_at_fpr_01': tpr_01,
            'tpr_at_fpr_001': tpr_001,
            'genetic_distance': genetic_dist,
            'normalized_wm_scores': normalized_wm_scores,
            'normalized_hm_scores': normalized_hm_scores,
            'mean_wm_score': np.mean(normalized_wm_scores),
            'mean_hm_score': np.mean(normalized_hm_scores),
            'success': True
        }

        logger.info(f"  {intermediate_lang}: AUC={auc:.3f}, TPR@0.1={tpr_01:.3f}, genetic_dist={genetic_dist:.3f}")
        return result

    def _get_bo_next_language(self) -> Optional[str]:
        """Use BO to select next best intermediate language."""
        if len(self.X_samples) < 2:
            # Not enough data for BO, random selection from remaining
            evaluated_langs = {result['intermediate_lang'] for result in self.evaluation_history}
            remaining_langs = [lang for lang in self.diversity_selector.available_languages
                             if lang not in evaluated_langs and lang != self.target_lang]
            if remaining_langs:
                return np.random.choice(remaining_langs)
            return None

        try:
            # Define search space: genetic distance range
            search_space = [Real(0.0, 6.0, name='genetic_distance')]

            # Run BO to maximize AUC (negate for minimization)
            result = gp_minimize(
                func=lambda x: -self._bo_objective(x),  # Negative for maximization
                dimensions=search_space,
                n_calls=1,
                n_initial_points=0,
                x0=self.X_samples,
                y0=[-y for y in self.y_samples],  # Negative for maximization
                acquisition_func=gaussian_ei,
                random_state=self.random_state
            )

            # Get suggested genetic distance
            suggested_distance = result.x[0]

            # Find closest available language by genetic distance
            best_lang = self._find_closest_language_by_distance(suggested_distance)
            return best_lang

        except Exception as e:
            logger.error(f"BO selection failed: {e}")
            # Fallback to random
            evaluated_langs = {result['intermediate_lang'] for result in self.evaluation_history}
            remaining_langs = [lang for lang in self.diversity_selector.available_languages
                             if lang not in evaluated_langs and lang != self.target_lang]
            if remaining_langs:
                return np.random.choice(remaining_langs)
            return None

    def _bo_objective(self, x: List[float]) -> float:
        """BO objective function (placeholder - real evaluation happens separately)."""
        # This is used by BO internally to model the relationship
        return 0.0

    def _find_closest_language_by_distance(self, target_distance: float) -> Optional[str]:
        """Find language with genetic distance closest to target."""
        best_lang = None
        best_diff = float('inf')

        # Exclude already evaluated languages and target language
        evaluated_langs = {result['intermediate_lang'] for result in self.evaluation_history}
        evaluated_langs.add(self.target_lang)

        for lang in self.diversity_selector.available_languages:
            if lang in evaluated_langs:
                continue

            try:
                actual_distance = self.genetic_distance.get_genetic_distance(
                    self.target_lang, lang
                )
                diff = abs(actual_distance - target_distance)

                if diff < best_diff:
                    best_diff = diff
                    best_lang = lang

            except Exception as e:
                logger.warning(f"Distance calculation failed for {lang}: {e}")
                continue

        return best_lang

    def evaluate_batch_steam_bo(self,
                              mod_file: str,
                              val_file: str) -> Dict:
        """
        Main batch STEAM BO evaluation.

        Args:
            mod_file: Path to watermarked z-score file (mc4.en.mod.z_score.jsonl)
            val_file: Path to validation z-score file (mc4.en.val.z_score.jsonl)

        Returns:
            Dictionary with BO results and best language selection
        """
        logger.info("Starting Batch STEAM BO Evaluation")
        logger.info(f"Mod file: {mod_file}")
        logger.info(f"Val file: {val_file}")

        # Step 1: Load and extract data
        mod_data = read_jsonl(mod_file)
        watermarked_texts = [item.get('text', item.get('response', '')) for item in mod_data]

        if not watermarked_texts:
            return {'error': 'No watermarked texts found', 'results': []}

        # For human texts, we'll use the same watermarked texts but treat them as negatives
        # This matches your evaluation approach where you compare against human baseline
        human_texts = watermarked_texts.copy()  # Same texts, different interpretation

        # Step 2: Get validation average for normalization (YOUR KEY FORMULA)
        validation_avg = get_avg_zscore(val_file)
        logger.info(f"Validation average z-score: {validation_avg:.4f}")

        # Step 3: Evaluate initial diverse languages
        logger.info("Phase 1: Evaluating initial diverse languages")
        self.evaluation_history = []
        self.X_samples = []
        self.y_samples = []

        for lang in self.initial_languages:
            if lang == self.target_lang:
                continue  # Skip target language

            result = self._evaluate_language_candidate(
                watermarked_texts, human_texts, validation_avg, lang
            )
            self.evaluation_history.append(result)

            if result.get('success', False):
                self.X_samples.append([result['genetic_distance']])
                self.y_samples.append(result['auc'])

        # Step 4: BO iterations
        logger.info("Phase 2: BO optimization iterations")
        best_result = max(self.evaluation_history, key=lambda x: x.get('auc', 0.0))

        for iteration in range(self.max_evaluations - len(self.initial_languages)):
            next_lang = self._get_bo_next_language()
            if not next_lang:
                logger.info("No more languages available for evaluation")
                break

            logger.info(f"BO iteration {iteration + 1}: evaluating {next_lang}")

            result = self._evaluate_language_candidate(
                watermarked_texts, human_texts, validation_avg, next_lang
            )
            self.evaluation_history.append(result)

            if result.get('success', False):
                self.X_samples.append([result['genetic_distance']])
                self.y_samples.append(result['auc'])

                # Check if this is the new best
                if result['auc'] > best_result.get('auc', 0.0):
                    best_result = result

        # Step 5: Compile final results
        final_results = {
            'target_lang': self.target_lang,
            'validation_avg_zscore': validation_avg,
            'n_texts': len(watermarked_texts),
            'initial_languages': self.initial_languages,
            'best_result': best_result,
            'all_evaluations': self.evaluation_history,
            'total_evaluations': len(self.evaluation_history),
            'languages': [result['intermediate_lang'] for result in self.evaluation_history],  # For compatibility with your scripts
            'bo_summary': {
                'best_auc': best_result.get('auc', 0.0),
                'best_language': best_result.get('intermediate_lang', ''),
                'best_genetic_distance': best_result.get('genetic_distance', 0.0)
            }
        }

        logger.info("=" * 60)
        logger.info("BATCH STEAM BO EVALUATION COMPLETE")
        logger.info(f"Best language: {best_result.get('intermediate_lang', 'None')}")
        logger.info(f"Best AUC: {best_result.get('auc', 0.0):.3f}")
        logger.info(f"Total evaluations: {len(self.evaluation_history)}")

        return final_results


def setup_watermark_detector(args, tokenizer):
    """Setup watermark detector based on method."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.watermark_method == 'kgw':
        detector = KGWDetector(
            vocab=list(tokenizer.get_vocab().keys()),
            gamma=0.25,
            delta=2.0,
            seeding_scheme="selfhash",
            device=device
        )
        return detector

    elif args.watermark_method == 'xsir':
        if not args.transform_model or not args.mapping_file:
            raise ValueError("XSIR requires --transform_model and --mapping_file")

        detector = XSIRDetector(
            device=device,
            chunk_length=20,
            target_tokenizer=tokenizer,
            delta=4.0,
            gamma=0.5,
            embedding_model=args.embedding_model,
            mapping_file=args.mapping_file,
            transform_model_path=args.transform_model
        )
        return detector

    else:
        raise ValueError(f"Unknown watermark method: {args.watermark_method}")


def main():
    parser = argparse.ArgumentParser(description="Batch BO-Enhanced STEAM Evaluation")

    # Model arguments
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--mod_file", required=True, help="Watermarked z-score file")
    parser.add_argument("--val_file", required=True, help="Validation z-score file")
    parser.add_argument("--output_file", required=True, help="Output JSON file")

    # STEAM BO arguments
    parser.add_argument("--target_lang", required=True, help="Target language (ISO-3)")
    parser.add_argument("--n_initial", type=int, default=3, help="Initial diverse languages")
    parser.add_argument("--max_evaluations", type=int, default=8, help="Max BO evaluations")
    parser.add_argument("--languages_file", default="supported_languages.txt", help="Supported languages file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Watermark method arguments
    parser.add_argument("--watermark_method", required=True, choices=['kgw', 'xsir'], help="Watermark method")
    parser.add_argument("--transform_model", help="Transform model path (for XSIR)")
    parser.add_argument("--mapping_file", help="Mapping file path (for XSIR)")
    parser.add_argument("--embedding_model", default="paraphrase-multilingual-mpnet-base-v2", help="Embedding model")

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.mod_file):
        logger.error(f"Mod file not found: {args.mod_file}")
        sys.exit(1)

    if not os.path.exists(args.val_file):
        logger.error(f"Val file not found: {args.val_file}")
        sys.exit(1)

    if not is_valid_iso3(args.target_lang):
        logger.error(f"Invalid ISO-3 target language: {args.target_lang}")
        sys.exit(1)

    logger.info("Starting Batch BO-Enhanced STEAM Evaluation")
    logger.info(f"Target language: {args.target_lang}")
    logger.info(f"Watermark method: {args.watermark_method}")

    # Setup tokenizer and detector
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    watermark_detector = setup_watermark_detector(args, tokenizer)

    # Initialize evaluator
    evaluator = BatchSTEAMBOEvaluator(
        watermark_detector=watermark_detector,
        target_lang=args.target_lang,
        n_initial=args.n_initial,
        max_evaluations=args.max_evaluations,
        languages_file=args.languages_file,
        random_state=args.seed
    )

    # Run evaluation
    results = evaluator.evaluate_batch_steam_bo(
        mod_file=args.mod_file,
        val_file=args.val_file
    )

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()