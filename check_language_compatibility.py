#!/usr/bin/env python3
"""
Check compatibility between URIEL languages and deep_translator supported languages.

This script analyzes which URIEL languages (ISO 639-3) can be mapped to
deep_translator supported languages (ISO 639-1) for the BO optimization.
"""

from language_code_converter import iso3_to_iso1, is_valid_iso3

# Deep translator supported languages (ISO 639-1 codes)
DEEP_TRANSLATOR_LANGUAGES = {
    'afrikaans': 'af',
    'albanian': 'sq',
    'amharic': 'am',
    'arabic': 'ar',
    'armenian': 'hy',
    'assamese': 'as',
    'aymara': 'ay',
    'azerbaijani': 'az',
    'bambara': 'bm',
    'basque': 'eu',
    'belarusian': 'be',
    'bengali': 'bn',
    'bhojpuri': 'bho',
    'bosnian': 'bs',
    'bulgarian': 'bg',
    'catalan': 'ca',
    'cebuano': 'ceb',
    'chichewa': 'ny',
    'chinese (simplified)': 'zh-CN',
    'chinese (traditional)': 'zh-TW',
    'corsican': 'co',
    'croatian': 'hr',
    'czech': 'cs',
    'danish': 'da',
    'dhivehi': 'dv',
    'dogri': 'doi',
    'dutch': 'nl',
    'english': 'en',
    'esperanto': 'eo',
    'estonian': 'et',
    'ewe': 'ee',
    'filipino': 'tl',
    'finnish': 'fi',
    'french': 'fr',
    'frisian': 'fy',
    'galician': 'gl',
    'georgian': 'ka',
    'german': 'de',
    'greek': 'el',
    'guarani': 'gn',
    'gujarati': 'gu',
    'haitian creole': 'ht',
    'hausa': 'ha',
    'hawaiian': 'haw',
    'hebrew': 'iw',
    'hindi': 'hi',
    'hmong': 'hmn',
    'hungarian': 'hu',
    'icelandic': 'is',
    'igbo': 'ig',
    'ilocano': 'ilo',
    'indonesian': 'id',
    'irish': 'ga',
    'italian': 'it',
    'japanese': 'ja',
    'javanese': 'jw',
    'kannada': 'kn',
    'kazakh': 'kk',
    'khmer': 'km',
    'kinyarwanda': 'rw',
    'konkani': 'gom',
    'korean': 'ko',
    'krio': 'kri',
    'kurdish (kurmanji)': 'ku',
    'kurdish (sorani)': 'ckb',
    'kyrgyz': 'ky',
    'lao': 'lo',
    'latin': 'la',
    'latvian': 'lv',
    'lingala': 'ln',
    'lithuanian': 'lt',
    'luganda': 'lg',
    'luxembourgish': 'lb',
    'macedonian': 'mk',
    'maithili': 'mai',
    'malagasy': 'mg',
    'malay': 'ms',
    'malayalam': 'ml',
    'maltese': 'mt',
    'maori': 'mi',
    'marathi': 'mr',
    'meiteilon (manipuri)': 'mni-Mtei',
    'mizo': 'lus',
    'mongolian': 'mn',
    'myanmar': 'my',
    'nepali': 'ne',
    'norwegian': 'no',
    'odia (oriya)': 'or',
    'oromo': 'om',
    'pashto': 'ps',
    'persian': 'fa',
    'polish': 'pl',
    'portuguese': 'pt',
    'punjabi': 'pa',
    'quechua': 'qu',
    'romanian': 'ro',
    'russian': 'ru',
    'samoan': 'sm',
    'sanskrit': 'sa',
    'scots gaelic': 'gd',
    'sepedi': 'nso',
    'serbian': 'sr',
    'sesotho': 'st',
    'shona': 'sn',
    'sindhi': 'sd',
    'sinhala': 'si',
    'slovak': 'sk',
    'slovenian': 'sl',
    'somali': 'so',
    'spanish': 'es',
    'sundanese': 'su',
    'swahili': 'sw',
    'swedish': 'sv',
    'tajik': 'tg',
    'tamil': 'ta',
    'tatar': 'tt',
    'telugu': 'te',
    'thai': 'th',
    'tigrinya': 'ti',
    'tsonga': 'ts',
    'turkish': 'tr',
    'turkmen': 'tk',
    'twi': 'ak',
    'ukrainian': 'uk',
    'urdu': 'ur',
    'uyghur': 'ug',
    'uzbek': 'uz',
    'vietnamese': 'vi',
    'welsh': 'cy',
    'xhosa': 'xh',
    'yiddish': 'yi',
    'yoruba': 'yo',
    'zulu': 'zu'
}

# Extract just the ISO codes
DEEP_TRANSLATOR_ISO_CODES = set(DEEP_TRANSLATOR_LANGUAGES.values())

def load_uriel_languages():
    """Load URIEL languages from all_languages.txt"""
    with open('all_languages.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

def check_compatibility():
    """Check which URIEL languages are compatible with deep_translator"""

    uriel_languages = load_uriel_languages()

    compatible_languages = []
    incompatible_languages = []

    print(f"Total URIEL languages: {len(uriel_languages)}")
    print(f"Total deep_translator languages: {len(DEEP_TRANSLATOR_ISO_CODES)}")
    print("="*60)

    for uriel_lang in uriel_languages:
        try:
            # Try to convert URIEL ISO 639-3 to ISO 639-1
            iso1_code = iso3_to_iso1(uriel_lang)

            # Check if this ISO 639-1 code is supported by deep_translator
            if iso1_code in DEEP_TRANSLATOR_ISO_CODES:
                compatible_languages.append((uriel_lang, iso1_code))
                print(f"✓ {uriel_lang} → {iso1_code} (supported)")
            else:
                incompatible_languages.append((uriel_lang, iso1_code, "not in deep_translator"))
                print(f"✗ {uriel_lang} → {iso1_code} (NOT supported by deep_translator)")

        except ValueError as e:
            incompatible_languages.append((uriel_lang, None, str(e)))
            print(f"✗ {uriel_lang} → ERROR: {e}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Compatible languages: {len(compatible_languages)}")
    print(f"Incompatible languages: {len(incompatible_languages)}")
    print(f"Compatibility rate: {len(compatible_languages)/len(uriel_languages)*100:.1f}%")

    # Save compatible languages for BO
    print(f"\nSaving compatible languages to 'supported_languages.txt'...")
    with open('supported_languages.txt', 'w') as f:
        for uriel_code, iso1_code in compatible_languages:
            f.write(f"{uriel_code}\n")

    print(f"\nCompatible URIEL codes for BO:")
    compatible_uriel_codes = [lang[0] for lang in compatible_languages]
    print(compatible_uriel_codes)

    # Show some problematic cases
    print(f"\nSome incompatible languages:")
    for lang, iso1, error in incompatible_languages[:10]:
        print(f"  {lang}: {error}")

    return compatible_languages, incompatible_languages

if __name__ == "__main__":
    compatible, incompatible = check_compatibility()