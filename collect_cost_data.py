#!/usr/bin/env python3
"""
Script to collect all 17-language back-translation cost data from JSON files
and prepare it for comprehensive analysis.
"""

import json
import os
from pathlib import Path
import sys

def get_language_full_name(lang_code):
    """Map language codes to full names."""
    language_map = {
        'bn': 'Bengali',
        'de': 'German',
        'es': 'Spanish',
        'fa': 'Persian',
        'fr': 'French',
        'hi': 'Hindi',
        'it': 'Italian',
        'iw': 'Hebrew',
        'ja': 'Japanese',
        'ko': 'Korean',
        'nl': 'Dutch',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ta': 'Tamil',
        'uk': 'Ukrainian',
        'vi': 'Vietnamese',
        'en': 'English'
    }
    return language_map.get(lang_code, lang_code.upper())

def load_all_cost_data():
    """Load all cost data from JSON files."""
    base_dir = Path('/Users/asim-abdalla/projects/ammi-research-project/watermarking/X-SIR-Re-Implementation/data/costs_17lang')

    models_data = {}

    for model_dir in ['llama-3.2-1B', 'aya-23-8B']:
        models_data[model_dir] = {}
        json_files = list((base_dir / model_dir / 'kgw_seed0').glob('*.json'))

        for json_file in json_files:
            # Extract language code from filename: mc4.en-XX_back_translation_cost.json
            filename = json_file.name
            if filename.startswith('mc4.en-') and filename.endswith('_back_translation_cost.json'):
                lang_code = filename.replace('mc4.en-', '').replace('_back_translation_cost.json', '')

                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        models_data[model_dir][lang_code] = data
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")

    return models_data

def format_number(num):
    """Format numbers with commas for readability."""
    return f"{num:,}"

def calculate_tokens_per_sample(tokens, samples):
    """Calculate tokens per sample."""
    return round(tokens / samples, 1) if samples > 0 else 0

def print_summary_data():
    """Print all collected data in organized format."""
    data = load_all_cost_data()

    print("=== COMPREHENSIVE 17-LANGUAGE BACK-TRANSLATION COST ANALYSIS ===\n")

    # Print raw data for verification
    for model_name, model_data in data.items():
        print(f"=== {model_name.upper()} ===")
        for lang_code, cost_data in sorted(model_data.items()):
            lang_full = get_language_full_name(lang_code)
            print(f"\n{lang_full} ({lang_code}):")
            print(f"  Input Tokens: {format_number(cost_data['input_tokens'])}")
            print(f"  Input Samples: {format_number(cost_data['input_samples'])}")
            print(f"  Output Tokens: {format_number(cost_data['output_tokens'])}")
            print(f"  Output Samples: {format_number(cost_data['output_samples'])}")
            print(f"  Total Tokens: {format_number(cost_data['total_tokens'])}")
            print(f"  Cost Multiplier: {cost_data['cost_multiplier']:.3f}")
            print(f"  Back-translation Files: {cost_data['back_translation_files']}")

            print("  Breakdown:")
            for item in cost_data['breakdown']:
                src_lang = get_language_full_name(item['source_lang'])
                tgt_lang = get_language_full_name(item['target_lang'])
                tokens_per_sample = calculate_tokens_per_sample(item['tokens'], item['samples'])
                print(f"    {src_lang} → {tgt_lang}: {format_number(item['tokens'])} tokens, {item['samples']} samples ({tokens_per_sample} tokens/sample)")
        print("\n" + "="*80 + "\n")

    return data

if __name__ == '__main__':
    try:
        collected_data = print_summary_data()
        print("Data collection completed successfully!")

        # Save collected data as JSON for the markdown generation
        with open('/Users/asim-abdalla/projects/ammi-research-project/watermarking/X-SIR-Re-Implementation/collected_cost_data.json', 'w') as f:
            json.dump(collected_data, f, indent=2)
        print("Data saved to collected_cost_data.json")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)