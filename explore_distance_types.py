#!/usr/bin/env python3

import urielplus
import numpy as np
import pandas as pd

def explore_distance_types():
    """Explore all available distance types in URIEL+"""
    u = urielplus.URIELPlus()

    # Test languages from different families
    test_languages = ['eng', 'spa', 'fra', 'deu', 'rus', 'jpn', 'zho', 'ara', 'hin']

    print("=== URIEL+ Distance Types Exploration ===\n")

    # Get all methods that contain 'distance'
    distance_methods = [method for method in dir(u) if 'distance' in method.lower()]
    print(f"Found {len(distance_methods)} distance-related methods:")
    for method in distance_methods:
        print(f"  - {method}")
    print()

    # Test each distance type
    distance_types = [
        ('Geographic', 'new_geographic_distance'),
        ('Genetic', 'new_genetic_distance'),
        ('Syntactic', 'new_syntactic_distance'),
        ('Phonological', 'new_phonological_distance'),
        ('Inventory', 'new_inventory_distance'),
        ('Featural', 'new_featural_distance'),
        ('Morphological', 'new_morphological_distance')
    ]

    results = {}

    for dist_name, method_name in distance_types:
        print(f"\n=== {dist_name} Distance ===")

        if hasattr(u, method_name):
            method = getattr(u, method_name)

            # Test a few language pairs
            test_pairs = [
                ('eng', 'spa'),  # English-Spanish
                ('eng', 'deu'),  # English-German
                ('eng', 'jpn'),  # English-Japanese
                ('spa', 'fra'),  # Spanish-French
                ('jpn', 'zho'),  # Japanese-Chinese
                ('ara', 'hin')   # Arabic-Hindi
            ]

            distances = {}
            for lang1, lang2 in test_pairs:
                try:
                    distance = method([lang1, lang2])
                    distances[f"{lang1}-{lang2}"] = distance
                    print(f"  {lang1} - {lang2}: {distance:.4f}")
                except Exception as e:
                    print(f"  {lang1} - {lang2}: ERROR - {e}")
                    distances[f"{lang1}-{lang2}"] = None

            results[dist_name] = distances
        else:
            print(f"  Method {method_name} not found")
            results[dist_name] = {}

    # Create comparison table
    print("\n=== Distance Comparison Table ===")
    df_data = []
    for pair in test_pairs:
        pair_name = f"{pair[0]}-{pair[1]}"
        row = {'Language_Pair': pair_name}
        for dist_type in [dt[0] for dt in distance_types]:
            if dist_type in results and pair_name in results[dist_type]:
                row[dist_type] = results[dist_type][pair_name]
            else:
                row[dist_type] = None
        df_data.append(row)

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))

    # Save results
    df.to_csv('distance_types_comparison.csv', index=False)
    print(f"\nResults saved to distance_types_comparison.csv")

    return results

def explain_distance_types():
    """Explain what each distance type represents linguistically"""

    explanations = {
        'Geographic': {
            'description': 'Physical distance between regions where languages are spoken',
            'measures': 'Spatial proximity of language communities',
            'interpretation': 'Lower values = languages spoken in nearby regions',
            'example': 'Spanish-Portuguese have low distance (both Iberian Peninsula)'
        },
        'Genetic': {
            'description': 'Historical relationship and common ancestry of languages',
            'measures': 'How languages evolved from common proto-languages',
            'interpretation': 'Lower values = more closely related in language family tree',
            'example': 'Spanish-French have low distance (both Romance languages)'
        },
        'Syntactic': {
            'description': 'Similarity in sentence structure and grammatical rules',
            'measures': 'Word order, case systems, agreement patterns, etc.',
            'interpretation': 'Lower values = similar grammatical structures',
            'example': 'English-German might have moderate distance (both Germanic but different word orders)'
        },
        'Phonological': {
            'description': 'Similarity in sound systems and pronunciation patterns',
            'measures': 'Consonants, vowels, phonotactics, stress patterns',
            'interpretation': 'Lower values = similar sound inventories and patterns',
            'example': 'Spanish-Italian have low distance (similar vowel systems)'
        },
        'Inventory': {
            'description': 'Similarity in phoneme inventories (speech sounds)',
            'measures': 'Which consonants and vowels exist in each language',
            'interpretation': 'Lower values = share more speech sounds',
            'example': 'Languages with similar consonant/vowel systems'
        },
        'Featural': {
            'description': 'Similarity in phonological features and distinctive features',
            'measures': 'Feature specifications of phonemes (voiced, nasal, etc.)',
            'interpretation': 'Lower values = phonemes have similar feature patterns',
            'example': 'Languages that use similar distinctive feature systems'
        },
        'Morphological': {
            'description': 'Similarity in word formation and inflectional patterns',
            'measures': 'Prefixes, suffixes, word-building strategies, inflection',
            'interpretation': 'Lower values = similar ways of forming/changing words',
            'example': 'Turkish-Hungarian might have low distance (both agglutinative)'
        }
    }

    print("\n=== Distance Type Explanations ===\n")

    for dist_type, info in explanations.items():
        print(f"📍 {dist_type} Distance")
        print(f"   Description: {info['description']}")
        print(f"   Measures: {info['measures']}")
        print(f"   Interpretation: {info['interpretation']}")
        print(f"   Example: {info['example']}")
        print()

if __name__ == "__main__":
    explain_distance_types()
    results = explore_distance_types()