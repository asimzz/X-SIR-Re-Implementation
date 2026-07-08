#!/usr/bin/env python3
import random
import json
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def select_random_languages(languages_file, n_initial=3, output_file='selected_languages.json'):
    """Select initial random languages (2-letter codes) and save to file."""
    from language_code_converter import iso3_to_iso1

    with open(languages_file, 'r') as f:
        all_langs = [line.strip() for line in f if line.strip()]

    # Convert 3-letter codes to 2-letter codes
    two_letter_langs = []
    for lang in all_langs:
        iso1_code = iso3_to_iso1(lang)
        if iso1_code:
            two_letter_langs.append(iso1_code)

    selected = random.sample(two_letter_langs, n_initial)

    with open(output_file, 'w') as f:
        json.dump(selected, f)

    print(f"Selected {n_initial} languages: {selected}")
    return selected

if __name__ == "__main__":
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'selected_languages.json'
    select_random_languages('supported_languages.txt', output_file=output_file)