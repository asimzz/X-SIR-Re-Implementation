#!/usr/bin/env python3

import pandas as pd

def create_cluster_analysis():
    """Create enhanced cluster analysis with language families"""

    # Load the clusters
    clusters_df = pd.read_csv('clusters_genetic.csv')

    # Language code to full name mapping
    language_names = {
        'eng': 'English', 'spa': 'Spanish', 'fra': 'French', 'deu': 'German',
        'ita': 'Italian', 'por': 'Portuguese', 'rus': 'Russian', 'jpn': 'Japanese',
        'zho': 'Chinese', 'ara': 'Arabic', 'hin': 'Hindi', 'kor': 'Korean',
        'nld': 'Dutch', 'pol': 'Polish', 'tur': 'Turkish', 'vie': 'Vietnamese',
        'tha': 'Thai', 'swe': 'Swedish', 'nor': 'Norwegian', 'dan': 'Danish',
        'fin': 'Finnish', 'hun': 'Hungarian', 'ces': 'Czech', 'slk': 'Slovak',
        'ron': 'Romanian', 'bul': 'Bulgarian', 'hrv': 'Croatian', 'srp': 'Serbian',
        'slv': 'Slovenian', 'est': 'Estonian', 'lav': 'Latvian', 'lit': 'Lithuanian',
        'ell': 'Greek', 'heb': 'Hebrew', 'fas': 'Persian', 'urd': 'Urdu',
        'ben': 'Bengali', 'tam': 'Tamil', 'tel': 'Telugu', 'kan': 'Kannada',
        'mal': 'Malayalam', 'guj': 'Gujarati', 'pan': 'Punjabi', 'mar': 'Marathi',
        'ori': 'Odia', 'asm': 'Assamese', 'nep': 'Nepali', 'sin': 'Sinhala',
        'mya': 'Myanmar', 'khm': 'Khmer', 'lao': 'Lao', 'mon': 'Mongolian',
        'bod': 'Tibetan', 'uig': 'Uyghur', 'kaz': 'Kazakh', 'kir': 'Kyrgyz',
        'uzb': 'Uzbek', 'tgk': 'Tajik', 'aze': 'Azerbaijani', 'kat': 'Georgian',
        'hye': 'Armenian', 'bel': 'Belarusian', 'ukr': 'Ukrainian', 'ltz': 'Luxembourgish',
        'mlt': 'Maltese', 'isl': 'Icelandic', 'fao': 'Faroese', 'gle': 'Irish',
        'gla': 'Scottish Gaelic', 'cym': 'Welsh', 'bre': 'Breton', 'eus': 'Basque',
        'cat': 'Catalan', 'glg': 'Galician', 'ast': 'Asturian', 'mwl': 'Mirandese',
        'vec': 'Venetian', 'lij': 'Ligurian', 'pms': 'Piedmontese', 'lmo': 'Lombard',
        'cos': 'Corsican', 'srd': 'Sardinian', 'scn': 'Sicilian', 'nap': 'Neapolitan',
        'lad': 'Ladino', 'arg': 'Aragonese', 'ext': 'Extremaduran', 'mdf': 'Moksha',
        'myv': 'Erzya', 'kpv': 'Komi-Zyrian', 'udm': 'Udmurt', 'krl': 'Karelian',
        'vep': 'Veps', 'izh': 'Ingrian', 'liv': 'Livonian', 'sme': 'Northern Sami',
        'smn': 'Inari Sami', 'sms': 'Skolt Sami', 'smj': 'Lule Sami', 'sma': 'Southern Sami',
        'afr': 'Afrikaans', 'swa': 'Swahili', 'hau': 'Hausa', 'yor': 'Yoruba',
        'ibo': 'Igbo', 'amh': 'Amharic', 'som': 'Somali', 'orm': 'Oromo'
    }

    # Language family analysis for each cluster
    def analyze_cluster_family(cluster_languages):
        """Determine the most likely language family for a cluster"""

        # Define language families
        families = {
            'Indo-European: Romance': ['spa', 'fra', 'ita', 'por', 'ron', 'cat', 'glg', 'ast', 'mwl', 'vec', 'lij', 'pms', 'lmo', 'cos', 'srd', 'scn', 'nap', 'lad', 'arg', 'ext'],
            'Indo-European: Germanic': ['eng', 'deu', 'nld', 'swe', 'nor', 'dan', 'ltz', 'isl', 'fao', 'afr'],
            'Indo-European: Slavic': ['rus', 'pol', 'ces', 'slk', 'bul', 'hrv', 'srp', 'slv', 'bel', 'ukr'],
            'Indo-European: Indo-Iranian': ['hin', 'fas', 'urd', 'ben', 'guj', 'pan', 'mar', 'ori', 'asm', 'nep', 'sin', 'tgk'],
            'Indo-European: Celtic': ['gle', 'gla', 'cym', 'bre'],
            'Indo-European: Baltic': ['lav', 'lit'],
            'Indo-European: Greek': ['ell'],
            'Indo-European: Armenian': ['hye'],
            'Uralic: Finno-Ugric': ['fin', 'hun', 'est', 'mdf', 'myv', 'kpv', 'udm', 'krl', 'vep', 'izh', 'liv', 'sme', 'smn', 'sms', 'smj', 'sma'],
            'Altaic: Turkic': ['tur', 'uig', 'kaz', 'kir', 'uzb', 'aze'],
            'Sino-Tibetan': ['zho', 'mya', 'bod'],
            'Afro-Asiatic: Semitic': ['ara', 'heb', 'mlt', 'amh'],
            'Afro-Asiatic: Cushitic': ['som', 'orm'],
            'Niger-Congo': ['swa', 'yor', 'ibo', 'hau'],
            'Austro-Asiatic': ['vie', 'khm'],
            'Tai-Kadai': ['tha', 'lao'],
            'Mongolic': ['mon'],
            'Kartvelian': ['kat'],
            'Language Isolate': ['jpn', 'kor', 'eus'],
            'Dravidian': ['tam', 'tel', 'kan', 'mal']
        }

        # Count matches for each family
        family_scores = {}
        for family, family_langs in families.items():
            score = sum(1 for lang in cluster_languages if lang in family_langs)
            if score > 0:
                family_scores[family] = score

        if not family_scores:
            return "Mixed/Unknown"

        # Get the family with highest score
        best_family = max(family_scores, key=family_scores.get)
        total_langs = len(cluster_languages)
        best_score = family_scores[best_family]

        # If family covers most languages in cluster, return it
        if best_score >= total_langs * 0.6:  # 60% threshold
            return best_family
        elif len(family_scores) == 1:
            return best_family
        else:
            # Mixed cluster - show main families
            sorted_families = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_families) >= 2:
                return f"Mixed: {sorted_families[0][0]} + {sorted_families[1][0]}"
            else:
                return best_family

    # Create enhanced dataframe
    enhanced_data = []

    for _, row in clusters_df.iterrows():
        lang_code = row['language']
        cluster_id = row['cluster_id']

        # Get language name
        lang_name = language_names.get(lang_code, lang_code.upper())

        enhanced_data.append({
            'cluster_id': cluster_id,
            'language_code': lang_code,
            'language_name': lang_name,
            'clustering_method': row['clustering_method']
        })

    enhanced_df = pd.DataFrame(enhanced_data)

    # Analyze each cluster's language family
    cluster_families = {}
    for cluster_id in enhanced_df['cluster_id'].unique():
        cluster_langs = enhanced_df[enhanced_df['cluster_id'] == cluster_id]['language_code'].tolist()
        family = analyze_cluster_family(cluster_langs)
        cluster_families[cluster_id] = family

    # Add language family column
    enhanced_df['language_family'] = enhanced_df['cluster_id'].map(cluster_families)

    # Reorder columns
    enhanced_df = enhanced_df[['cluster_id', 'language_code', 'language_name', 'language_family', 'clustering_method']]

    # Sort by cluster_id, then by language_name
    enhanced_df = enhanced_df.sort_values(['cluster_id', 'language_name'])

    # Save to CSV
    enhanced_df.to_csv('clusters_analysis.csv', index=False)

    print(f"✅ Created enhanced analysis: clusters_analysis.csv")
    print(f"📊 Analyzed {len(enhanced_df)} languages in {enhanced_df['cluster_id'].nunique()} clusters")

    # Print summary
    print(f"\n=== Cluster Family Analysis ===")
    for cluster_id in sorted(enhanced_df['cluster_id'].unique()):
        cluster_data = enhanced_df[enhanced_df['cluster_id'] == cluster_id]
        family = cluster_data.iloc[0]['language_family']
        langs = cluster_data['language_name'].tolist()
        print(f"Cluster {cluster_id:2d} ({len(langs):2d}L): {family}")
        print(f"           Languages: {', '.join(langs)}")
        print()

    return enhanced_df

if __name__ == "__main__":
    df = create_cluster_analysis()