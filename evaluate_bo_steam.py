#!/usr/bin/env python3
"""
BO-Enhanced STEAM Watermark Evaluation with URIEL Genetic Distances

This script evaluates watermark detection using Bayesian Optimization to select
the best intermediate language for each text sample. Uses real URIEL genetic
distances (NO CLUSTERS).

Usage:
    python evaluate_bo_steam.py \
        --base_model meta-llama/Llama-3.2-1B \
        --input_file data/translated.jsonl \
        --output_file results/bo_results.jsonl \
        --target_lang fr \
        --watermark_method kgw \
        --n_initial 3 \
        --max_evaluations 8

Author: Asim
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

# Import our BO-enhanced detector
from steam_bo_detector import SteamBOPerTextDetector
from language_code_converter import iso1_to_iso3, is_valid_iso1, is_valid_iso3

# Import watermark detectors (assume these are available in the project)
try:
    from src_watermark.kgw import KGWDetector
    from src_watermark.xsir import XSIRDetector
except ImportError:
    print("⚠️  Warning: Watermark detector modules not found. Using mock detectors for testing.")
    
    class KGWDetector:
        def __init__(self, **kwargs):
            pass
        
        def detect(self, text: str) -> Dict[str, float]:
            # Mock implementation for testing
            return {"z_score": np.random.uniform(-2, 5)}
    
    class XSIRDetector:
        def __init__(self, **kwargs):
            pass
        
        def detect(self, text: str) -> Dict[str, float]:
            # Mock implementation for testing
            return {"z_score": np.random.uniform(-2, 5)}


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: Dict, file_path: str):
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def initialize_watermark_detector(args):
    """Initialize the appropriate watermark detector."""
    if args.watermark_method == 'kgw':
        logger.info("Initializing KGW watermark detector")
        detector = KGWDetector(
            base_model=args.base_model,
            seed=args.seed
        )
    elif args.watermark_method == 'xsir':
        logger.info("Initializing XSIR watermark detector")
        if not args.transform_model or not args.mapping_file:
            raise ValueError("XSIR requires --transform_model and --mapping_file")
        
        detector = XSIRDetector(
            base_model=args.base_model,
            transform_model=args.transform_model,
            embedding_model=args.embedding_model,
            mapping_file=args.mapping_file,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown watermark method: {args.watermark_method}")
    
    return detector


def compute_summary_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from BO results."""
    z_scores = [r['z_score'] for r in results if r['success']]
    evaluations = [r['total_evaluations'] for r in results if r['success']]
    
    if not z_scores:
        return {
            'n_samples': len(results),
            'n_successful': 0,
            'error': 'No successful detections'
        }
    
    # Count language selections
    language_counts = {}
    for r in results:
        if r['success']:
            lang = r['best_intermediate_lang']
            language_counts[lang] = language_counts.get(lang, 0) + 1
    
    summary = {
        'n_samples': len(results),
        'n_successful': len(z_scores),
        'n_failed': len(results) - len(z_scores),
        
        # Z-score statistics
        'z_score_mean': float(np.mean(z_scores)),
        'z_score_std': float(np.std(z_scores)),
        'z_score_median': float(np.median(z_scores)),
        'z_score_min': float(np.min(z_scores)),
        'z_score_max': float(np.max(z_scores)),
        
        # Evaluation statistics
        'evaluations_mean': float(np.mean(evaluations)),
        'evaluations_std': float(np.std(evaluations)),
        'evaluations_median': float(np.median(evaluations)),
        
        # Language selection frequency
        'language_selection_counts': language_counts,
        'most_selected_language': max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else None,
        
        # Detection statistics (assuming threshold of 4.0)
        'detection_rate_z4': float(sum(1 for z in z_scores if z >= 4.0) / len(z_scores)),
        'detection_rate_z3': float(sum(1 for z in z_scores if z >= 3.0) / len(z_scores)),
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='BO-Enhanced STEAM Watermark Evaluation')
    
    # Input/output
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSONL file with translated texts')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSONL file for detailed results')
    parser.add_argument('--summary_file', type=str, required=True,
                        help='Output JSON file for summary statistics')
    
    # Model configuration
    parser.add_argument('--base_model', type=str, required=True,
                        help='Base model name')
    parser.add_argument('--target_lang', type=str, required=True,
                        help='Target language code (ISO 639-1)')
    
    # Watermark method
    parser.add_argument('--watermark_method', type=str, required=True,
                        choices=['kgw', 'xsir'],
                        help='Watermark detection method')
    
    # XSIR-specific arguments
    parser.add_argument('--transform_model', type=str,
                        help='Path to XSIR transform model (required for XSIR)')
    parser.add_argument('--embedding_model', type=str, default='paraphrase-multilingual-mpnet-base-v2',
                        help='Embedding model for XSIR')
    parser.add_argument('--mapping_file', type=str,
                        help='Path to XSIR mapping file (required for XSIR)')
    
    # BO configuration
    parser.add_argument('--n_initial', type=int, default=3,
                        help='Number of initial random language samples')
    parser.add_argument('--max_evaluations', type=int, default=8,
                        help='Maximum number of language evaluations per text')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    
    # Other
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--languages_file', type=str, default='all_languages.txt',
                        help='File containing list of intermediate languages')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    data = load_jsonl(args.input_file)
    
    # Limit to n_samples if specified
    if args.n_samples is not None:
        data = data[:args.n_samples]
        logger.info(f"Limited to {args.n_samples} samples")
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Initialize watermark detector
    watermark_detector = initialize_watermark_detector(args)
    
    # Initialize BO-enhanced STEAM detector
    logger.info("Initializing BO-enhanced STEAM detector with URIEL genetic distances")
    bo_detector = SteamBOPerTextDetector(
        watermark_detector=watermark_detector,
        languages_file=args.languages_file,
        max_evaluations=args.max_evaluations,
        initial_random_samples=args.n_initial
    )
    
    logger.info(f"Using {len(bo_detector.all_languages)} intermediate languages")
    logger.info(f"Target language (input): {args.target_lang}")
    
    # Convert target language to ISO 639-3 if needed (for URIEL compatibility)
    if len(args.target_lang) == 2 and is_valid_iso1(args.target_lang):
        # Convert ISO 639-1 to ISO 639-3
        target_lang_iso3 = iso1_to_iso3(args.target_lang)
        logger.info(f"Converted {args.target_lang} (ISO 639-1) → {target_lang_iso3} (ISO 639-3)")
    elif len(args.target_lang) == 3 and is_valid_iso3(args.target_lang):
        # Already ISO 639-3
        target_lang_iso3 = args.target_lang
        logger.info(f"Using ISO 639-3 code: {target_lang_iso3}")
    else:
        # Unknown format, try as-is and hope for the best
        logger.warning(f"Could not validate language code: {args.target_lang}")
        logger.warning("Attempting to use as-is...")
        target_lang_iso3 = args.target_lang
    
    logger.info(f"BO config: {args.n_initial} initial + {args.max_evaluations - args.n_initial} BO iterations")
    
    # Run BO-enhanced STEAM evaluation
    results = []
    failed_count = 0
    
    logger.info("Starting BO-enhanced STEAM evaluation...")
    for item in tqdm(data, desc=f"Evaluating {args.target_lang}"):
        # Get the translated text
        text = item.get('text', item.get('translation', ''))
        
        if not text:
            logger.warning(f"Empty text in sample {item.get('id', 'unknown')}")
            failed_count += 1
            continue
        
        try:
            # Run BO-enhanced detection
            detection_result = bo_detector.detect_with_bo(
                text=text,
                target_lang=target_lang_iso3
            )
            
            # Store results
            result = {
                'id': item.get('id', len(results)),
                'text': text[:100] + '...' if len(text) > 100 else text,  # Truncate for storage
                'target_lang': args.target_lang,
                'best_intermediate_lang': detection_result.best_intermediate_lang,
                'z_score': detection_result.z_score,
                'total_evaluations': detection_result.total_evaluations,
                'success': detection_result.success,
                'error': detection_result.error,
                'evaluation_history': detection_result.evaluation_history,
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            failed_count += 1
            results.append({
                'id': item.get('id', len(results)),
                'success': False,
                'error': str(e)
            })
    
    # Compute summary statistics
    logger.info("Computing summary statistics...")
    summary = compute_summary_statistics(results)
    summary['config'] = {
        'base_model': args.base_model,
        'watermark_method': args.watermark_method,
        'target_lang': args.target_lang,
        'n_initial': args.n_initial,
        'max_evaluations': args.max_evaluations,
        'n_samples': len(data),
        'seed': args.seed,
        'n_failed': failed_count,
    }
    
    # Save results
    logger.info(f"Saving detailed results to {args.output_file}")
    save_jsonl(results, args.output_file)
    
    logger.info(f"Saving summary to {args.summary_file}")
    save_json(summary, args.summary_file)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("BO-ENHANCED STEAM EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Target Language: {args.target_lang}")
    logger.info(f"Watermark Method: {args.watermark_method}")
    logger.info(f"Samples: {summary['n_successful']}/{summary['n_samples']} successful")
    logger.info(f"Z-score: {summary['z_score_mean']:.3f} ± {summary['z_score_std']:.3f}")
    logger.info(f"Detection rate (z≥4): {summary['detection_rate_z4']:.1%}")
    logger.info(f"Detection rate (z≥3): {summary['detection_rate_z3']:.1%}")
    logger.info(f"Avg evaluations: {summary['evaluations_mean']:.1f}")
    if summary.get('most_selected_language'):
        logger.info(f"Most selected language: {summary['most_selected_language']}")
    logger.info("="*50)
    
    logger.info("✅ BO-enhanced STEAM evaluation complete!")


if __name__ == '__main__':
    main()