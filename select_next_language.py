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

def compute_mean_zscore(mod_file, val_file):
    """Compute mean normalized z-score from mod file."""
    with open(mod_file, 'r') as f:
        mod_data = [json.loads(line) for line in f if line.strip()]

    with open(val_file, 'r') as f:
        val_data = [json.loads(line) for line in f if line.strip()]

    mod_scores = [d['z_score'] for d in mod_data]
    val_scores = [d['z_score'] for d in val_data]

    # Validation-based normalization
    avg_val = np.mean(val_scores)
    normalized_mod_scores = [s - avg_val for s in mod_scores]

    return np.mean(normalized_mod_scores)

def select_next_language(target_lang, out_dir, languages_file, history_file='bo_history.json', selected_file='selected_languages.json', max_evaluations=8):
    """Use BO to select next best language."""
    from language_code_converter import iso1_to_iso3, iso3_to_iso1

    # Load available languages and convert 3-letter to 2-letter codes
    with open(languages_file, 'r') as f:
        raw_languages = [line.strip() for line in f if line.strip()]

    all_languages = []
    for lang in raw_languages:
        if len(lang) == 3:  # 3-letter code, convert to 2-letter
            iso1_code = iso3_to_iso1(lang)
            if iso1_code:
                all_languages.append(iso1_code)
        elif len(lang) == 2:  # Already 2-letter
            all_languages.append(lang)

    target_lang_iso3 = iso1_to_iso3(target_lang)

    # Get evaluated languages and their performance
    evaluated_langs = []
    zscores = []
    genetic_dists = []

    # Load evaluation history
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
            evaluated_langs = history['languages']
            zscores = history['zscores']
            genetic_dists = history['genetic_distances']
    except FileNotFoundError:
        # First iteration - collect initial evaluations
        for lang in json.load(open(selected_file)):
            mod_file = f"{out_dir}/mc4.{target_lang}-{lang}-back.mod.z_score.jsonl"
            val_file = f"{out_dir}/mc4.{target_lang}-{lang}-back.val.z_score.jsonl"

            try:
                mean_zscore = compute_mean_zscore(mod_file, val_file)
                lang_iso3 = iso1_to_iso3(lang)
                genetic_dist = get_genetic_distance(target_lang_iso3, lang_iso3)

                evaluated_langs.append(lang)
                zscores.append(mean_zscore)
                genetic_dists.append(genetic_dist)
            except:
                continue

    if len(evaluated_langs) == 0:
        return None

    # Save current history
    history = {
        'languages': evaluated_langs,
        'zscores': zscores,
        'genetic_distances': genetic_dists
    }
    with open(history_file, 'w') as f:
        json.dump(history, f)

    # Skip if budget exhausted
    if len(evaluated_langs) >= max_evaluations:
        best_idx = np.argmax(zscores)  # Best = highest z-score
        return evaluated_langs[best_idx]

    # Prepare for BO optimization
    X = np.array(genetic_dists).reshape(-1, 1)
    y = [-zscore for zscore in zscores]  # Minimize negative z-score = Maximize z-score

    # Get available genetic distances for remaining languages (no filtering yet)
    remaining_langs = [l for l in all_languages if l not in evaluated_langs]

    candidate_dists = []
    candidate_langs = []

    for lang in remaining_langs:
        try:
            lang_iso3 = iso1_to_iso3(lang)
            dist = get_genetic_distance(target_lang_iso3, lang_iso3)
            candidate_dists.append(dist)
            candidate_langs.append(lang)
        except:
            continue

    if len(candidate_langs) == 0:
        best_idx = np.argmax(zscores)
        return evaluated_langs[best_idx]

    # Proper BO optimization
    if len(X) >= 2:
        # Define search space - include existing distances in bounds
        all_dists = genetic_dists + candidate_dists
        min_dist = min(all_dists)
        max_dist = max(all_dists)
        space = [Real(min_dist, max_dist)]

        # BO to find optimal genetic distance
        def objective(genetic_distance):
            """Evaluate a genetic distance by finding closest language and computing z-score."""
            target_dist = genetic_distance[0]

            # Find language closest to this genetic distance
            best_lang = None
            min_dist_diff = float('inf')

            for i, lang in enumerate(candidate_langs):
                dist_diff = abs(candidate_dists[i] - target_dist)
                if dist_diff < min_dist_diff:
                    min_dist_diff = dist_diff
                    best_lang = lang

            if best_lang is None:
                return 0.0  # Fallback

            # Get z-score for this language (simulate evaluation)
            # In real implementation, this would translate and detect
            lang_iso3 = iso1_to_iso3(best_lang)
            actual_dist = get_genetic_distance(target_lang_iso3, lang_iso3)

            # For now, use interpolation from existing data to estimate z-score
            if len(X) >= 2:
                from scipy.interpolate import interp1d
                interp_func = interp1d(X.flatten(), [-yi for yi in y],
                                     kind='linear', fill_value='extrapolate')
                estimated_zscore = interp_func(actual_dist)
                return -estimated_zscore  # Minimize negative = maximize z-score
            else:
                return 0.0

        # Need at least 10 calls for BO, use existing data + new evaluations
        n_calls = max(10, len(X) + 3)

        result = gp_minimize(
            func=objective,
            dimensions=space,
            x0=X.tolist(),
            y0=y,
            n_calls=n_calls,
            random_state=0
        )

        # Get suggested optimal distance
        optimal_dist = result.x[0]

        # Find language closest to optimal distance
        distances_to_optimal = [abs(d - optimal_dist) for d in candidate_dists]
        selected_idx = np.argmin(distances_to_optimal)
        selected_lang = candidate_langs[selected_idx]

    else:
        # Random selection if not enough data
        selected_idx = np.random.randint(0, len(candidate_langs))
        selected_lang = candidate_langs[selected_idx]

    # Check if deep_translator supports this language
    from deep_translator import GoogleTranslator
    supported_codes = set(GoogleTranslator().get_supported_languages(as_dict=True).values())

    if selected_lang in supported_codes:
        # Update history with selected language (placeholder AUC until evaluated)
        selected_lang_iso3 = iso1_to_iso3(selected_lang)
        selected_dist = get_genetic_distance(target_lang_iso3, selected_lang_iso3)

        evaluated_langs.append(selected_lang)
        zscores.append(0.0)  # Placeholder z-score
        genetic_dists.append(selected_dist)

        # Save updated history
        history = {
            'languages': evaluated_langs,
            'zscores': zscores,
            'genetic_distances': genetic_dists
        }
        with open(history_file, 'w') as f:
            json.dump(history, f)

        return selected_lang
    else:

        # Filter candidates to only supported languages
        supported_candidates = []
        supported_dists = []

        for i, lang in enumerate(candidate_langs):
            if lang in supported_codes:
                supported_candidates.append(lang)
                supported_dists.append(candidate_dists[i])

        if len(supported_candidates) > 0:
            if len(X) >= 2:
                # Find closest to optimal distance among supported languages
                distances_to_optimal = [abs(d - optimal_dist) for d in supported_dists]
                selected_idx = np.argmin(distances_to_optimal)
                selected_lang = supported_candidates[selected_idx]

                # Update history with fallback selection
                selected_lang_iso3 = iso1_to_iso3(selected_lang)
                selected_dist = get_genetic_distance(target_lang_iso3, selected_lang_iso3)

                evaluated_langs.append(selected_lang)
                zscores.append(0.0)  # Placeholder z-score
                genetic_dists.append(selected_dist)

                # Save updated history
                history = {
                    'languages': evaluated_langs,
                    'zscores': zscores,
                    'genetic_distances': genetic_dists
                }
                with open(history_file, 'w') as f:
                    json.dump(history, f)

                return selected_lang
            else:
                # Random from supported
                selected_idx = np.random.randint(0, len(supported_candidates))
                selected_lang = supported_candidates[selected_idx]

                # Update history with random selection
                selected_lang_iso3 = iso1_to_iso3(selected_lang)
                selected_dist = get_genetic_distance(target_lang_iso3, selected_lang_iso3)

                evaluated_langs.append(selected_lang)
                zscores.append(0.0)  # Placeholder z-score
                genetic_dists.append(selected_dist)

                # Save updated history
                history = {
                    'languages': evaluated_langs,
                    'zscores': zscores,
                    'genetic_distances': genetic_dists
                }
                with open(history_file, 'w') as f:
                    json.dump(history, f)

                return selected_lang
        else:
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_lang', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--languages_file', default='supported_languages.txt')
    parser.add_argument('--history_file', default='bo_history.json')
    parser.add_argument('--selected_file', default='selected_languages.json')
    parser.add_argument('--max_evaluations', type=int, default=8)
    args = parser.parse_args()

    next_lang = select_next_language(args.target_lang, args.out_dir, args.languages_file, args.history_file, args.selected_file, args.max_evaluations)
    if next_lang:
        print(next_lang)
    else:
        print("ERROR")