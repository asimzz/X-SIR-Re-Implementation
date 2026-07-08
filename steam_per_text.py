#!/usr/bin/env python3
import json
import numpy as np
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

def detect_watermark_single(text, base_model, watermark_method, seed, **kwargs):
    """Detect watermark on a single text and return z-score."""
    import tempfile
    import subprocess

    # Create temporary file with single text
    temp_data = [{"response": text}]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in temp_data:
            f.write(json.dumps(item) + '\n')
        temp_input = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_output = f.name

    # Build detection command
    cmd = [
        'python3', 'detect.py',
        '--base_model', base_model,
        '--seed', str(seed),
        '--detect_file', temp_input,
        '--output_file', temp_output,
        '--watermark_method', watermark_method
    ]

    # Add additional args based on watermark method
    if watermark_method == 'xsir':
        if 'transform_model' in kwargs:
            cmd.extend(['--transform_model', kwargs['transform_model']])
        if 'embedding_model' in kwargs:
            cmd.extend(['--embedding_model', kwargs['embedding_model']])
        if 'mapping_file' in kwargs:
            cmd.extend(['--mapping_file', kwargs['mapping_file']])

    try:
        subprocess.run(cmd, capture_output=True, check=True)

        # Read result
        with open(temp_output, 'r') as f:
            result = json.loads(f.readline())
            return result.get('z_score', 0.0)
    except:
        return 0.0
    finally:
        os.unlink(temp_input)
        os.unlink(temp_output)

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

def compute_validation_zscore_for_language(target_lang, intermediate_lang, base_model, watermark_method, seed, val_file_path=None, **detect_kwargs):
    """Compute average validation z-score for a specific intermediate language using all validation texts."""
    val_texts = load_validation_file(target_lang, val_file_path)

    if len(val_texts) == 0:
        return 0.0

    validation_zscores = []

    # Process validation texts through STEAM path
    for val_text in val_texts:
        # en → target_lang → intermediate_lang → en (STEAM path)
        step1 = translate_text(val_text, 'en', target_lang)
        step2 = translate_text(step1, target_lang, intermediate_lang)
        back_translated = translate_text(step2, intermediate_lang, 'en')

        # Detect watermark on back-translated validation text
        zscore = detect_watermark_single(back_translated, base_model, watermark_method, seed, **detect_kwargs)
        validation_zscores.append(zscore)

    # Return average validation z-score for this language
    return np.mean(validation_zscores)

def steam_per_text(text_item, target_lang, base_model, watermark_method, seed,
                   n_initial=3, max_evaluations=8, val_file_path=None, is_watermarked=True, **detect_kwargs):
    """Run STEAM BO optimization for a single text with validation normalization (only for watermarked texts)."""
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
        # en → target_lang → lang → en (STEAM back-translation)
        step1 = translate_text(original_text, 'en', target_lang)
        step2 = translate_text(step1, target_lang, lang)
        back_translated = translate_text(step2, lang, 'en')

        # Detect watermark on back-translated text
        raw_zscore = detect_watermark_single(back_translated, base_model, watermark_method, seed, **detect_kwargs)

        if is_watermarked:
            # Get validation z-score for normalization (only for watermarked texts)
            val_zscore = compute_validation_zscore_for_language(target_lang, lang, base_model, watermark_method, seed, val_file_path, **detect_kwargs)
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
        step1 = translate_text(original_text, 'en', target_lang)
        step2 = translate_text(step1, target_lang, next_lang)
        back_translated = translate_text(step2, next_lang, 'en')

        raw_zscore = detect_watermark_single(back_translated, base_model, watermark_method, seed, **detect_kwargs)

        if is_watermarked:
            # Get validation z-score for normalization (only for watermarked texts)
            val_zscore = compute_validation_zscore_for_language(target_lang, next_lang, base_model, watermark_method, seed, val_file_path, **detect_kwargs)
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
    parser = argparse.ArgumentParser(description="STEAM per-text BO optimization")
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

    # Load input data
    with open(args.input_file, 'r') as f:
        texts = [json.loads(line) for line in f if line.strip()]

    print(f"Processing {len(texts)} texts with STEAM per-text BO...")

    results = []

    # Process each text independently
    for i, text_item in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}")

        detect_kwargs = {}
        if args.transform_model:
            detect_kwargs['transform_model'] = args.transform_model
        if args.embedding_model:
            detect_kwargs['embedding_model'] = args.embedding_model
        if args.mapping_file:
            detect_kwargs['mapping_file'] = args.mapping_file

        result = steam_per_text(
            text_item=text_item,
            target_lang=args.target_lang,
            base_model=args.base_model,
            watermark_method=args.watermark_method,
            seed=args.seed,
            n_initial=args.n_initial,
            max_evaluations=args.max_evaluations,
            val_file_path=args.val_file,
            is_watermarked=args.is_watermarked,
            **detect_kwargs
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