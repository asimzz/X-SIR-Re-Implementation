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
import torch
from transformers import AutoTokenizer

# Import our corrected STEAM BO detector
from steam_bo_detector import STEAMBODetector
from language_code_converter import iso1_to_iso3, is_valid_iso1, is_valid_iso3


from src_watermark.kgw.extended_watermark_processor import WatermarkDetector as KGWDetector
from src_watermark.xsir.watermark import WatermarkContext as XSIRDetector


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
    import torch
    import numpy as np

    def convert_to_serializable(obj):
        """Convert torch Tensors and numpy arrays to Python types."""
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            serializable_item = convert_to_serializable(item)
            f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')


def save_json(data: Dict, file_path: str):
    """Save data to JSON file."""
    import torch
    import numpy as np

    def convert_to_serializable(obj):
        """Convert torch Tensors and numpy arrays to Python types."""
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj

    with open(file_path, 'w', encoding='utf-8') as f:
        serializable_data = convert_to_serializable(data)
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


def initialize_watermark_detector(args):
    """Initialize the appropriate watermark detector."""
    # Setup device and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    if args.watermark_method == 'kgw':
        logger.info("Initializing KGW watermark detector")
        
        # Set defaults for KGW parameters if not provided
        gamma = getattr(args, 'gamma', 0.25)
        seeding_scheme = getattr(args, 'seeding_scheme', 'minhash')
        
        detector = KGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            seed=args.seed,
            seeding_scheme=seeding_scheme,
            device=device,
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
        
    elif args.watermark_method == 'xsir':
        logger.info("Initializing XSIR watermark detector")
        if not args.transform_model or not args.mapping_file:
            raise ValueError("XSIR requires --transform_model and --mapping_file")
        
        # Set defaults for XSIR parameters
        chunk_size = getattr(args, 'chunk_size', 10)
        delta = getattr(args, 'delta', 1.0)
        
        detector = XSIRDetector(
            device=device,
            chunk_size=chunk_size,
            tokenizer=tokenizer,
            mapping_file=args.mapping_file,
            delta=delta,
            transform_model_path=args.transform_model,
            embedding_model=args.embedding_model,
        )
    else:
        raise ValueError(f"Unknown watermark method: {args.watermark_method}")
    
    return detector


def compute_summary_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from STEAM BO results."""
    successful_results = [r for r in results if r['success']]

    if not successful_results:
        return {
            'n_samples': len(results),
            'n_successful': 0,
            'error': 'No successful detections'
        }

    # Extract normalized and raw z-scores
    normalized_z_scores = [r['normalized_z_score'] for r in successful_results]
    raw_z_scores = [r['raw_z_score'] for r in successful_results]
    evaluations = [r['total_evaluations'] for r in successful_results]
    genetic_distances = [r['best_genetic_distance'] for r in successful_results]

    # Count language selections
    language_counts = {}
    for r in successful_results:
        lang = r['best_intermediate_lang']
        language_counts[lang] = language_counts.get(lang, 0) + 1

    # Genetic distance analysis
    distance_bins = np.histogram(genetic_distances, bins=5)

    summary = {
        'n_samples': len(results),
        'n_successful': len(successful_results),
        'n_failed': len(results) - len(successful_results),

        # Normalized z-score statistics
        'normalized_z_score_mean': float(np.mean(normalized_z_scores)),
        'normalized_z_score_std': float(np.std(normalized_z_scores)),
        'normalized_z_score_median': float(np.median(normalized_z_scores)),
        'normalized_z_score_min': float(np.min(normalized_z_scores)),
        'normalized_z_score_max': float(np.max(normalized_z_scores)),

        # Raw z-score statistics
        'raw_z_score_mean': float(np.mean(raw_z_scores)),
        'raw_z_score_std': float(np.std(raw_z_scores)),
        'raw_z_score_median': float(np.median(raw_z_scores)),
        'raw_z_score_min': float(np.min(raw_z_scores)),
        'raw_z_score_max': float(np.max(raw_z_scores)),

        # Evaluation statistics
        'evaluations_mean': float(np.mean(evaluations)),
        'evaluations_std': float(np.std(evaluations)),
        'evaluations_median': float(np.median(evaluations)),

        # Genetic distance analysis
        'genetic_distance_mean': float(np.mean(genetic_distances)),
        'genetic_distance_std': float(np.std(genetic_distances)),
        'genetic_distance_median': float(np.median(genetic_distances)),
        'genetic_distance_distribution': {
            'bin_edges': distance_bins[1].tolist(),
            'bin_counts': distance_bins[0].tolist()
        },

        # Language selection frequency
        'language_selection_counts': language_counts,
        'most_selected_language': max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else None,

        # Detection statistics (both normalized and raw)
        'detection_rate_norm_z1': float(sum(1 for z in normalized_z_scores if z >= 1.0) / len(normalized_z_scores)),
        'detection_rate_norm_z2': float(sum(1 for z in normalized_z_scores if z >= 2.0) / len(normalized_z_scores)),
        'detection_rate_raw_z4': float(sum(1 for z in raw_z_scores if z >= 4.0) / len(raw_z_scores)),
        'detection_rate_raw_z3': float(sum(1 for z in raw_z_scores if z >= 3.0) / len(raw_z_scores)),
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
    parser.add_argument('--delta', type=float, default=None,
                        help='Delta parameter for XSIR (default: 1.0)')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='Chunk size for XSIR context window')
    
    # KGW-specific arguments
    parser.add_argument('--gamma', type=float, default=0.25,
                        help='Gamma parameter for KGW watermarking')
    parser.add_argument('--seeding_scheme', type=str, default='minhash',
                        help='Seeding scheme for KGW (default: minhash)')
    
    # BO configuration
    parser.add_argument('--n_initial', type=int, default=3,
                        help='Number of initial random language samples')
    parser.add_argument('--max_evaluations', type=int, default=8,
                        help='Maximum number of language evaluations per text')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--normalization_samples', type=int, default=50,
                        help='Number of samples for z-score normalization calibration')
    
    # Other
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--languages_file', type=str, default='supported_languages.txt',
                        help='File containing list of deep_translator-supported intermediate languages')
    
    args = parser.parse_args()
    
    # Set default delta values based on watermark method (matching user's code)
    if args.watermark_method == 'kgw' and args.delta is None:
        args.delta = 2.0
    elif args.watermark_method == 'xsir' and args.delta is None:
        args.delta = 1.0
    
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
    
    # Initialize STEAM BO detector with supervisor's approach
    logger.info("Initializing STEAM BO detector with supervisor's approach")
    steam_detector = STEAMBODetector(
        watermark_detector=watermark_detector,
        languages_file=args.languages_file,
        max_evaluations=args.max_evaluations,
        n_initial=args.n_initial,
        normalization_samples=args.normalization_samples,
        random_state=args.seed
    )
    
    logger.info(f"Using {len(steam_detector.all_languages)} intermediate languages")
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
    
    # Calibrate z-score normalization
    logger.info("Calibrating z-score normalization...")
    calibration_texts = []
    for item in data[:args.normalization_samples]:
        text = item.get('response', item.get('text', item.get('translation', '')))
        if text:
            calibration_texts.append(text)

    steam_detector.calibrate_z_score_normalization(calibration_texts, target_lang_iso3)

    logger.info(f"STEAM BO config: {args.n_initial} initial + {args.max_evaluations - args.n_initial} BO iterations")

    # Run STEAM BO evaluation
    results = []
    failed_count = 0
    
    logger.info("Starting BO-enhanced STEAM evaluation...")
    
    # Debug: Show available fields in first sample
    if len(data) > 0:
        sample_fields = list(data[0].keys())
        logger.info(f"Available fields in JSONL: {sample_fields}")
    
    for item in tqdm(data, desc=f"Evaluating {args.target_lang}"):
        # Get the translated text
        # The JSONL files have the translated text in 'response' field
        text = item.get('response', item.get('text', item.get('translation', '')))
        
        if not text:
            logger.warning(f"Empty text in sample {item.get('id', 'unknown')}")
            failed_count += 1
            continue
        
        try:
            # Run STEAM BO detection
            detection_result = steam_detector.detect_with_steam_bo(
                text=text,
                target_lang=target_lang_iso3
            )
            
            # Store results
            result = {
                'id': item.get('id', len(results)),
                'text': text[:100] + '...' if len(text) > 100 else text,  # Truncate for storage
                'target_lang': args.target_lang,
                'best_intermediate_lang': detection_result.best_intermediate_lang,
                'raw_z_score': detection_result.raw_z_score,
                'normalized_z_score': detection_result.normalized_z_score,
                'best_genetic_distance': detection_result.best_genetic_distance,
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
        'normalization_samples': args.normalization_samples,
        'n_samples': len(data),
        'seed': args.seed,
        'n_failed': failed_count,
        'z_score_calibration': {
            'mean': steam_detector.z_score_mean,
            'std': steam_detector.z_score_std,
            'is_calibrated': steam_detector.is_calibrated
        }
    }
    
    # Save results
    logger.info(f"Saving detailed results to {args.output_file}")
    save_jsonl(results, args.output_file)
    
    logger.info(f"Saving summary to {args.summary_file}")
    save_json(summary, args.summary_file)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("STEAM BO EVALUATION SUMMARY (Supervisor's Approach)")
    logger.info("="*60)
    logger.info(f"Target Language: {args.target_lang}")
    logger.info(f"Watermark Method: {args.watermark_method}")
    logger.info(f"Samples: {summary.get('n_successful', 0)}/{summary.get('n_samples', 0)} successful")
    logger.info(f"Z-score calibration: mean={steam_detector.z_score_mean:.3f}, std={steam_detector.z_score_std:.3f}")

    if summary.get('n_successful', 0) > 0:
        logger.info(f"Normalized z-score: {summary['normalized_z_score_mean']:.3f} ± {summary['normalized_z_score_std']:.3f}")
        logger.info(f"Raw z-score: {summary['raw_z_score_mean']:.3f} ± {summary['raw_z_score_std']:.3f}")
        logger.info(f"Detection rate (norm_z≥2): {summary['detection_rate_norm_z2']:.1%}")
        logger.info(f"Detection rate (raw_z≥4): {summary['detection_rate_raw_z4']:.1%}")
        logger.info(f"Avg evaluations: {summary['evaluations_mean']:.1f}")
        logger.info(f"Avg genetic distance: {summary['genetic_distance_mean']:.3f}")
        if summary.get('most_selected_language'):
            logger.info(f"Most selected language: {summary['most_selected_language']}")
    else:
        logger.error("❌ No successful samples - all texts failed processing")

    logger.info("="*60)
    logger.info("✅ STEAM BO evaluation complete!")


if __name__ == '__main__':
    main()