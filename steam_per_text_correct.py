#!/usr/bin/env python3
import json
import numpy as np
import torch
from skopt import gp_minimize
from skopt.space import Real
import argparse
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_genetic_distance(lang1, lang2):
    """Get genetic distance using URIEL."""
    from urielplus import urielplus
    try:
        uriel = urielplus.URIELPlus()
        distance = uriel.new_genetic_distance([lang1, lang2])
        return float(distance)
    except:
        return 0.5  # fallback

def translate_text(text, src_lang, tgt_lang):
    """Translate a single text using deep_translator."""
    from deep_translator import GoogleTranslator
    try:
        translator = GoogleTranslator(source=src_lang, target=tgt_lang)
        return translator.translate(text)
    except:
        return text  # fallback

def setup_watermark_detector(base_model, watermark_method, seed, **kwargs):
    """Setup watermark detector for per-text detection."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src_watermark.kgw.extended_watermark_processor import WatermarkDetector as KGWDetector
    from src_watermark.xsir.watermark import WatermarkWindow as XSIRWindow, WatermarkContext as XSIRContext

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if watermark_method == "kgw":
        detector = KGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=0.25,  # default gamma
            seed=seed,
            seeding_scheme="simple_1",  # default seeding scheme
            device=device,
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
    elif watermark_method in ["xsir", "sir"]:
        if kwargs.get('watermark_type', 'context') == "window":
            detector = XSIRWindow(
                device,
                kwargs.get('window_size', 1),
                tokenizer
            )
        else:  # context
            detector = XSIRContext(
                device,
                kwargs.get('chunk_size', 20),
                tokenizer,
                mapping_file=kwargs.get('mapping_file'),
                delta=kwargs.get('delta', 1.5),
                transform_model_path=kwargs.get('transform_model'),
                embedding_model=kwargs.get('embedding_model')
            )
    else:
        raise ValueError(f"Unsupported watermark method: {watermark_method}")

    return detector, tokenizer

def detect_watermark_single(text, detector):
    """Detect watermark on a single text and return z-score."""
    try:
        with torch.no_grad():
            result = detector.detect(text)
            z_score = result.get("z_score")
            if z_score is None or (z_score != z_score):  # check for NaN
                return 0.0
            return float(z_score)
    except:
        return 0.0

def load_validation_file(target_lang, val_file_path=None):
    """Load validation texts for target language."""
    if val_file_path is None:
        val_file = f"gen/llama-3.2-1B/kgw_seed0/mc4.en-{target_lang}.val.jsonl"
    else:
        val_file = val_file_path

    try:
        with open(val_file, 'r') as f:
            val_texts = [json.loads(line)['response'] for line in f if line.strip()]
        return val_texts
    except FileNotFoundError:
        print(f"Warning: Validation file not found: {val_file}")
        return []

def compute_validation_zscore_for_language(target_lang, selected_lang, detector, val_file_path=None):
    """Compute average validation z-score for a specific selected language using all validation texts."""
    val_texts = load_validation_file(target_lang, val_file_path)

    if len(val_texts) == 0:
        return 0.0

    validation_zscores = []

    # Process validation texts through STEAM path
    for val_text in val_texts:
        # en → target_lang → selected_lang (STOP here - correct path!)
        step1 = translate_text(val_text, 'en', target_lang)
        final_text = translate_text(step1, target_lang, selected_lang)

        # Detect watermark on final text (in selected_lang)
        zscore = detect_watermark_single(final_text, detector)
        validation_zscores.append(zscore)

    # Return average validation z-score for this language
    return np.mean(validation_zscores)

def steam_per_text_correct(text_item, target_lang, detector,
                          n_initial=3, max_evaluations=8, val_file_path=None, is_watermarked=True):
    """Run correct STEAM BO optimization for a single text."""
    from language_code_converter import iso1_to_iso3, iso3_to_iso1
    from deep_translator import GoogleTranslator

    # Get supported languages
    supported_codes = set(GoogleTranslator().get_supported_languages(as_dict=True).values())

    # Load available languages
    with open('supported_languages.txt', 'r') as f:
        raw_languages = [line.strip() for line in f if line.strip()]

    all_languages = []
    for lang in raw_languages:
        if len(lang) == 3:  # 3-letter to 2-letter
            iso1_code = iso3_to_iso1(lang)
            if iso1_code and iso1_code in supported_codes:
                all_languages.append(iso1_code)
        elif len(lang) == 2 and lang in supported_codes:
            all_languages.append(lang)

    # Remove target language
    all_languages = [l for l in all_languages if l != target_lang]

    if len(all_languages) < n_initial:
        return {"best_lang": None, "best_zscore": 0.0, "error": "Not enough languages"}

    # Get original text
    original_text = text_item['response']
    target_lang_iso3 = iso1_to_iso3(target_lang)

    # Step 1: Random initial languages
    import random
    random.seed(42)  # For reproducibility per text
    initial_langs = random.sample(all_languages, n_initial)

    evaluated_langs = []
    raw_zscores = []
    final_scores = []  # Raw for human, normalized for watermarked
    validation_zscores = []
    genetic_dists = []

    # Evaluate initial languages
    for lang in initial_langs:
        # CORRECT STEAM path: en → target_lang → lang (STOP here)
        step1 = translate_text(original_text, 'en', target_lang)
        final_text = translate_text(step1, target_lang, lang)

        # Detect watermark on final text (in selected language)
        raw_zscore = detect_watermark_single(final_text, detector)

        if is_watermarked:
            # Get validation z-score for normalization (only for watermarked texts)
            val_zscore = compute_validation_zscore_for_language(target_lang, lang, detector, val_file_path)
            final_score = raw_zscore - val_zscore  # Normalized z-score
        else:
            # For human texts, use raw z-score directly
            val_zscore = 0.0  # Not used for human texts
            final_score = raw_zscore  # Raw z-score

        # Get genetic distance
        lang_iso3 = iso1_to_iso3(lang)
        genetic_dist = get_genetic_distance(target_lang_iso3, lang_iso3)

        evaluated_langs.append(lang)
        raw_zscores.append(raw_zscore)
        final_scores.append(final_score)
        validation_zscores.append(val_zscore)
        genetic_dists.append(genetic_dist)

    # BO optimization loop
    remaining_langs = [l for l in all_languages if l not in evaluated_langs]

    for iteration in range(max_evaluations - n_initial):
        if len(remaining_langs) == 0:
            break

        # Prepare BO data (use final scores - raw for human, normalized for watermarked)
        X = np.array(genetic_dists).reshape(-1, 1)
        y = [-score for score in final_scores]  # Minimize negative = maximize

        # Get candidate distances
        candidate_dists = []
        candidate_langs = []

        for lang in remaining_langs:
            lang_iso3 = iso1_to_iso3(lang)
            dist = get_genetic_distance(target_lang_iso3, lang_iso3)
            candidate_dists.append(dist)
            candidate_langs.append(lang)

        if len(candidate_langs) == 0:
            break

        # BO optimization
        if len(X) >= 2:
            # Define search space
            all_dists = genetic_dists + candidate_dists
            min_dist = min(all_dists)
            max_dist = max(all_dists)
            space = [Real(min_dist, max_dist)]

            # BO to find optimal distance
            def objective(genetic_distance):
                target_dist = genetic_distance[0]

                # Find closest language to target distance
                best_lang = None
                min_dist_diff = float('inf')

                for i, lang in enumerate(candidate_langs):
                    dist_diff = abs(candidate_dists[i] - target_dist)
                    if dist_diff < min_dist_diff:
                        min_dist_diff = dist_diff
                        best_lang = lang

                if best_lang is None:
                    return 0.0

                # Estimate final score using interpolation
                if len(X) >= 2:
                    from scipy.interpolate import interp1d
                    lang_iso3 = iso1_to_iso3(best_lang)
                    actual_dist = get_genetic_distance(target_lang_iso3, lang_iso3)

                    interp_func = interp1d(X.flatten(), [-yi for yi in y],
                                         kind='linear', fill_value='extrapolate')
                    estimated_score = interp_func(actual_dist)
                    return -estimated_score
                else:
                    return 0.0

            n_calls = max(10, len(X) + 3)

            result = gp_minimize(
                func=objective,
                dimensions=space,
                x0=X.tolist(),
                y0=y,
                n_calls=n_calls,
                random_state=42
            )

            optimal_dist = result.x[0]

            # Find language closest to optimal distance
            distances_to_optimal = [abs(d - optimal_dist) for d in candidate_dists]
            selected_idx = np.argmin(distances_to_optimal)
            next_lang = candidate_langs[selected_idx]
        else:
            # Random selection
            next_lang = random.choice(candidate_langs)

        # Evaluate selected language
        # CORRECT STEAM path: en → target_lang → next_lang (STOP here)
        step1 = translate_text(original_text, 'en', target_lang)
        final_text = translate_text(step1, target_lang, next_lang)

        raw_zscore = detect_watermark_single(final_text, detector)

        if is_watermarked:
            # Get validation z-score for normalization (only for watermarked texts)
            val_zscore = compute_validation_zscore_for_language(target_lang, next_lang, detector, val_file_path)
            final_score = raw_zscore - val_zscore  # Normalized z-score
        else:
            # For human texts, use raw z-score directly
            val_zscore = 0.0  # Not used for human texts
            final_score = raw_zscore  # Raw z-score

        next_lang_iso3 = iso1_to_iso3(next_lang)
        genetic_dist = get_genetic_distance(target_lang_iso3, next_lang_iso3)

        evaluated_langs.append(next_lang)
        raw_zscores.append(raw_zscore)
        final_scores.append(final_score)
        validation_zscores.append(val_zscore)
        genetic_dists.append(genetic_dist)

        remaining_langs.remove(next_lang)

    # Return best result (based on final scores)
    best_idx = np.argmax(final_scores)
    return {
        "best_lang": evaluated_langs[best_idx],
        "best_zscore": final_scores[best_idx],  # Return final score (raw for human, normalized for watermarked)
        "best_raw_zscore": raw_zscores[best_idx],
        "best_validation_zscore": validation_zscores[best_idx] if is_watermarked else None,
        "is_watermarked": is_watermarked,
        "all_langs": evaluated_langs,
        "all_final_scores": final_scores,
        "all_raw_zscores": raw_zscores,
        "all_validation_zscores": validation_zscores if is_watermarked else None,
        "genetic_distances": genetic_dists
    }

def main():
    parser = argparse.ArgumentParser(description="Correct STEAM per-text BO optimization")
    parser.add_argument('--input_file', required=True, help='Input JSONL file')
    parser.add_argument('--output_file', required=True, help='Output results file')
    parser.add_argument('--target_lang', required=True, help='Target language')
    parser.add_argument('--base_model', required=True, help='Base model')
    parser.add_argument('--watermark_method', required=True, help='Watermark method')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_initial', type=int, default=3, help='Initial random languages')
    parser.add_argument('--max_evaluations', type=int, default=8, help='Max BO evaluations')

    # Watermark-specific args
    parser.add_argument('--transform_model', help='Transform model for X-SIR')
    parser.add_argument('--embedding_model', help='Embedding model for X-SIR')
    parser.add_argument('--mapping_file', help='Mapping file for X-SIR')
    parser.add_argument('--val_file', help='Validation file path')
    parser.add_argument('--is_watermarked', action='store_true', help='Whether input texts are watermarked (affects normalization)')

    args = parser.parse_args()

    # Setup detector
    detect_kwargs = {}
    if args.transform_model:
        detect_kwargs['transform_model'] = args.transform_model
    if args.embedding_model:
        detect_kwargs['embedding_model'] = args.embedding_model
    if args.mapping_file:
        detect_kwargs['mapping_file'] = args.mapping_file

    detector, tokenizer = setup_watermark_detector(args.base_model, args.watermark_method, args.seed, **detect_kwargs)

    # Load input data
    with open(args.input_file, 'r') as f:
        texts = [json.loads(line) for line in f if line.strip()]

    print(f"Processing {len(texts)} texts with correct STEAM per-text BO...")

    results = []

    # Process each text independently
    for i, text_item in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}")

        result = steam_per_text_correct(
            text_item=text_item,
            target_lang=args.target_lang,
            detector=detector,
            n_initial=args.n_initial,
            max_evaluations=args.max_evaluations,
            val_file_path=args.val_file,
            is_watermarked=args.is_watermarked
        )

        result['text_id'] = i
        results.append(result)

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    successful_results = [r for r in results if r.get('best_lang')]
    avg_zscore = np.mean([r['best_zscore'] for r in successful_results]) if successful_results else 0

    print(f"Completed! Average best z-score: {avg_zscore:.4f}")
    print(f"Successful optimizations: {len(successful_results)}/{len(texts)}")

if __name__ == "__main__":
    main()