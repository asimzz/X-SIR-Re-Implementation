import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer

# Import our batch STEAM BO detector
from steam_bo_batch_detector import STEAMBOBatchDetector
from language_code_converter import iso1_to_iso3, is_valid_iso1, is_valid_iso3

from src_watermark.kgw.extended_watermark_processor import WatermarkDetector as KGWDetector
from src_watermark.xsir.watermark import WatermarkContext as XSIRDetector


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_json(data: Dict, file_path: str):
    """Save data to JSON file with tensor conversion."""
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if args.watermark_method == 'kgw':
        logger.info("Initializing KGW watermark detector")

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


def compute_summary_statistics(result: Any) -> Dict[str, Any]:
    """Compute summary statistics from batch BO results."""
    if not result.success:
        return {
            'success': False,
            'error': result.error or 'Unknown error'
        }

    # Extract AUC scores from evaluation history
    auc_scores = [e['auc'] for e in result.evaluation_history if e['success']]
    genetic_distances = [e['genetic_distance'] for e in result.evaluation_history if e['success']]

    # Count language selections by iteration
    languages_tried = [(e['intermediate_lang'], e['auc']) for e in result.evaluation_history if e['success']]

    summary = {
        'success': True,
        'target_lang': result.target_lang,
        'best_intermediate_lang': result.best_intermediate_lang,
        'best_auc': result.best_auc,
        'best_genetic_distance': result.best_genetic_distance,
        'total_evaluations': result.total_evaluations,

        # AUC statistics across all evaluations
        'auc_mean': float(np.mean(auc_scores)) if auc_scores else 0.0,
        'auc_std': float(np.std(auc_scores)) if auc_scores else 0.0,
        'auc_min': float(np.min(auc_scores)) if auc_scores else 0.0,
        'auc_max': float(np.max(auc_scores)) if auc_scores else 0.0,

        # Genetic distance statistics
        'genetic_distance_mean': float(np.mean(genetic_distances)) if genetic_distances else 0.0,
        'genetic_distance_std': float(np.std(genetic_distances)) if genetic_distances else 0.0,
        'genetic_distance_range': [float(np.min(genetic_distances)), float(np.max(genetic_distances))] if genetic_distances else [0.0, 0.0],

        # Languages tried and their performance
        'languages_tried': languages_tried,
        'n_languages_evaluated': len(set([lang for lang, _ in languages_tried])),

        # Performance improvement
        'auc_improvement': result.best_auc - auc_scores[0] if len(auc_scores) > 0 else 0.0,

        # Full evaluation history
        'evaluation_history': result.evaluation_history
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Batch STEAM BO Evaluation (Real Supervisor Approach)')

    # Input files
    parser.add_argument('--mod_file', type=str, required=True,
                        help='Input JSONL file with watermarked samples (mc4.en-fa.mod.z_score.jsonl)')
    parser.add_argument('--val_file', type=str, required=True,
                        help='Input JSONL file with validation samples (mc4.en-fa.val.z_score.jsonl)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file for results')

    # Model configuration
    parser.add_argument('--base_model', type=str, required=True,
                        help='Base model name')
    parser.add_argument('--target_lang', type=str, required=True,
                        help='Target language code (ISO 639-1 or 639-3)')

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
                        help='Maximum number of language evaluations')

    # Other
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--languages_file', type=str, default='supported_languages.txt',
                        help='File containing list of deep_translator-supported intermediate languages')

    args = parser.parse_args()

    # Set default delta values based on watermark method
    if args.watermark_method == 'kgw' and args.delta is None:
        args.delta = 2.0
    elif args.watermark_method == 'xsir' and args.delta is None:
        args.delta = 1.0

    # Set random seed
    np.random.seed(args.seed)

    # Validate input files exist
    if not Path(args.mod_file).exists():
        logger.error(f"Watermarked file not found: {args.mod_file}")
        sys.exit(1)

    if not Path(args.val_file).exists():
        logger.error(f"Validation file not found: {args.val_file}")
        sys.exit(1)

    # Initialize watermark detector
    logger.info("Initializing watermark detector...")
    watermark_detector = initialize_watermark_detector(args)

    # Convert target language to ISO 639-3 if needed (for URIEL compatibility)
    if len(args.target_lang) == 2 and is_valid_iso1(args.target_lang):
        target_lang_iso3 = iso1_to_iso3(args.target_lang)
        logger.info(f"Converted {args.target_lang} (ISO 639-1) → {target_lang_iso3} (ISO 639-3)")
    elif len(args.target_lang) == 3 and is_valid_iso3(args.target_lang):
        target_lang_iso3 = args.target_lang
        logger.info(f"Using ISO 639-3 code: {target_lang_iso3}")
    else:
        logger.warning(f"Could not validate language code: {args.target_lang}")
        target_lang_iso3 = args.target_lang

    # Initialize batch STEAM BO detector
    logger.info("Initializing Batch STEAM BO detector with real supervisor's approach")
    batch_detector = STEAMBOBatchDetector(
        watermark_detector=watermark_detector,
        languages_file=args.languages_file,
        max_evaluations=args.max_evaluations,
        n_initial=args.n_initial,
        random_state=args.seed
    )

    logger.info(f"Using {len(batch_detector.all_languages)} intermediate languages")
    logger.info(f"Target language: {args.target_lang} → {target_lang_iso3}")

    logger.info(f"Batch BO config: {args.n_initial} initial + {args.max_evaluations - args.n_initial} BO iterations")
    logger.info(f"This will process ALL 500 samples for each language evaluation")

    # Run batch STEAM BO optimization
    logger.info("Starting batch STEAM BO optimization...")

    result = batch_detector.optimize_with_batch_bo(
        mod_file=args.mod_file,
        val_file=args.val_file,
        target_lang=target_lang_iso3
    )

    # Compute summary statistics
    logger.info("Computing summary statistics...")
    summary = compute_summary_statistics(result)
    summary['config'] = {
        'base_model': args.base_model,
        'watermark_method': args.watermark_method,
        'target_lang': args.target_lang,
        'target_lang_iso3': target_lang_iso3,
        'mod_file': args.mod_file,
        'val_file': args.val_file,
        'n_initial': args.n_initial,
        'max_evaluations': args.max_evaluations,
        'seed': args.seed,
    }

    # Save results
    logger.info(f"Saving results to {args.output_file}")
    save_json(summary, args.output_file)

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("BATCH STEAM BO EVALUATION SUMMARY (Real Supervisor's Approach)")
    logger.info("="*70)
    logger.info(f"Target Language: {args.target_lang} ({target_lang_iso3})")
    logger.info(f"Watermark Method: {args.watermark_method}")
    logger.info(f"Input files: {args.mod_file}, {args.val_file}")

    if summary['success']:
        logger.info(f"✅ Optimization successful!")
        logger.info(f"Best intermediate language: {summary['best_intermediate_lang']}")
        logger.info(f"Best AUC: {summary['best_auc']:.4f}")
        logger.info(f"Best genetic distance: {summary['best_genetic_distance']:.3f}")
        logger.info(f"Total evaluations: {summary['total_evaluations']}")
        logger.info(f"Languages evaluated: {summary['n_languages_evaluated']}")
        logger.info(f"AUC improvement: {summary['auc_improvement']:.4f}")

        # Show evaluation progression
        logger.info(f"\nEvaluation progression:")
        for i, (lang, auc) in enumerate(summary['languages_tried'], 1):
            logger.info(f"  {i}. {lang}: AUC = {auc:.4f}")

    else:
        logger.error(f"❌ Optimization failed: {summary.get('error', 'Unknown error')}")

    logger.info("="*70)
    logger.info("✅ Batch STEAM BO evaluation complete!")


if __name__ == '__main__':
    main()