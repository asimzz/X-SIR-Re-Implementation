#!/usr/bin/env python3
"""
BO-Enhanced STEAM Evaluation Script

This script evaluates the BO-enhanced STEAM watermark detection approach
against translated text, comparing it with the original exhaustive search method.
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import torch

from utils import read_jsonl, append_jsonl
from steam_bo_detector import SteamBODetector

# Import watermark detectors
from src_watermark.xsir.watermark import (
    WatermarkWindow as XSIRWindow,
    WatermarkContext as XSIRContext,
)
from src_watermark.kgw.extended_watermark_processor import (
    WatermarkLogitsProcessor as KGWLogitsProcessor
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_watermark_detector(args):
    """Load the appropriate watermark detector based on method."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.watermark_method == "kgw":
        print("Loading KGW watermark detector...")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # Create KGW detector
        detector = KGWLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=0.5,
            delta=2.0,
            seed=args.seed,
            seeding_scheme="simple_1",
        )

        return detector

    elif args.watermark_method == "xsir":
        print("Loading XSIR watermark detector...")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # Create XSIR detector
        detector = XSIRWindow(
            transform_model_path=args.transform_model,
            embedding_model=args.embedding_model,
            mapping_file_path=args.mapping_file,
            target_tokenizer=tokenizer,
            gamma=0.5,
            delta=2.0
        )

        return detector

    else:
        raise ValueError(f"Unknown watermark method: {args.watermark_method}")


def evaluate_single_text(text: str, target_lang: str, steam_detector: SteamBODetector) -> Dict[str, Any]:
    """
    Evaluate a single text using BO-enhanced STEAM.

    Args:
        text: Text to evaluate (in target language)
        target_lang: Target language code
        steam_detector: BO-enhanced STEAM detector instance

    Returns:
        Dictionary with evaluation results
    """
    start_time = time.time()

    # Run BO-enhanced detection
    result = steam_detector.detect_with_bo(text, target_lang)

    evaluation_time = time.time() - start_time

    return {
        'text': text,
        'target_lang': target_lang,
        'success': result.success,
        'best_z_score': result.z_score,
        'best_intermediate_lang': result.best_intermediate_lang,
        'total_evaluations': result.total_evaluations,
        'evaluation_time': evaluation_time,
        'evaluation_history': result.evaluation_history,
        'error': result.error
    }


def main():
    parser = argparse.ArgumentParser(description="BO-Enhanced STEAM Evaluation")

    # Model arguments
    parser.add_argument("--base_model", required=True, help="Base model name/path")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")

    # Input/Output arguments
    parser.add_argument("--input_file", required=True, help="Input JSONL file with translated text")
    parser.add_argument("--output_file", required=True, help="Output JSONL file for results")
    parser.add_argument("--summary_file", required=True, help="Output JSON file for summary stats")

    # BO configuration
    parser.add_argument("--target_lang", required=True, help="Target language code")
    parser.add_argument("--n_initial", type=int, default=3, help="Number of initial random samples")
    parser.add_argument("--max_evaluations", type=int, default=8, help="Maximum evaluations per text")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of text samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Watermark method arguments
    parser.add_argument("--watermark_method", required=True, choices=["kgw", "xsir"], help="Watermark method")
    parser.add_argument("--transform_model", help="XSIR transform model path")
    parser.add_argument("--embedding_model", help="XSIR embedding model")
    parser.add_argument("--mapping_file", help="XSIR mapping file")

    args = parser.parse_args()

    print(f"BO-Enhanced STEAM: {args.watermark_method} ({args.target_lang})")

    try:
        # Load watermark detector
        watermark_detector = load_watermark_detector(args)

        # Initialize BO-enhanced STEAM detector
        steam_detector = SteamBODetector(
            watermark_detector=watermark_detector,
            n_initial=args.n_initial,
            max_evaluations=args.max_evaluations,
            random_state=args.seed
        )

        # Load input data
        input_data = read_jsonl(args.input_file)
        if args.n_samples > 0:
            input_data = input_data[:args.n_samples]

        # Create output directory
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)

        # Evaluation loop
        results = []
        successful_evaluations = 0
        total_evaluation_time = 0
        total_language_evaluations = 0

        for i, sample in enumerate(input_data):
            try:
                text = sample.get('response', '')
                if not text:
                    continue

                result = evaluate_single_text(text, args.target_lang, steam_detector)
                results.append(result)

                if result['success']:
                    successful_evaluations += 1
                    total_evaluation_time += result['evaluation_time']
                    total_language_evaluations += result['total_evaluations']

            except Exception as e:
                continue

        # Save final results
        with open(args.output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        # Generate summary statistics
        summary = {
            'evaluation_config': {
                'target_lang': args.target_lang,
                'watermark_method': args.watermark_method,
                'n_initial': args.n_initial,
                'max_evaluations': args.max_evaluations,
                'n_samples_requested': args.n_samples,
                'seed': args.seed
            },
            'results': {
                'total_samples': len(input_data),
                'successful_evaluations': successful_evaluations,
                'success_rate': successful_evaluations / len(input_data) if input_data else 0,
                'avg_evaluation_time': total_evaluation_time / successful_evaluations if successful_evaluations > 0 else 0,
                'avg_language_evaluations': total_language_evaluations / successful_evaluations if successful_evaluations > 0 else 0,
                'total_language_evaluations': total_language_evaluations
            }
        }

        # Add z-score statistics
        if successful_evaluations > 0:
            z_scores = [r['best_z_score'] for r in results if r['success']]
            summary['results']['z_score_stats'] = {
                'mean': sum(z_scores) / len(z_scores),
                'min': min(z_scores),
                'max': max(z_scores),
                'count': len(z_scores)
            }

            # Language usage statistics
            lang_usage = {}
            for r in results:
                if r['success'] and r['best_intermediate_lang']:
                    lang = r['best_intermediate_lang']
                    lang_usage[lang] = lang_usage.get(lang, 0) + 1

            summary['results']['language_usage'] = lang_usage

        # Save summary
        with open(args.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Evaluated {successful_evaluations}/{len(input_data)} samples successfully")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()