"""
Language Code Conversion Utility

Converts between ISO 639-1 (2-letter) and ISO 639-3 (3-letter) codes.
Useful for bridging translation APIs (which often use ISO 639-1) with
URIEL database (which uses ISO 639-3).

Author: Asim
"""

# Mapping from ISO 639-1 (2-letter) to ISO 639-3 (3-letter)
# Based on common languages in watermark evaluation
ISO_639_1_TO_3 = {
    # High-resource languages
    'en': 'eng',  # English
    'fr': 'fra',  # French
    'de': 'deu',  # German
    'it': 'ita',  # Italian
    'es': 'spa',  # Spanish
    'pt': 'por',  # Portuguese
    
    # Medium-resource languages
    'pl': 'pol',  # Polish
    'nl': 'nld',  # Dutch
    'ru': 'rus',  # Russian
    'hi': 'hin',  # Hindi
    'ko': 'kor',  # Korean
    'ja': 'jpn',  # Japanese
    'zh': 'zho',  # Chinese (Mandarin)
    'ar': 'ara',  # Arabic
    
    # Low-resource languages
    'bn': 'ben',  # Bengali
    'fa': 'fas',  # Persian (Farsi)
    'vi': 'vie',  # Vietnamese
    'he': 'heb',  # Hebrew
    'iw': 'heb',  # Hebrew (old code)
    'uk': 'ukr',  # Ukrainian
    'ta': 'tam',  # Tamil
    'th': 'tha',  # Thai
    'tr': 'tur',  # Turkish
    'id': 'ind',  # Indonesian
    'ms': 'msa',  # Malay
    'sw': 'swa',  # Swahili
    'ro': 'ron',  # Romanian
    'cs': 'ces',  # Czech
    'sv': 'swe',  # Swedish
    'da': 'dan',  # Danish
    'no': 'nor',  # Norwegian
    'fi': 'fin',  # Finnish
    'el': 'ell',  # Greek
    'hu': 'hun',  # Hungarian
    'sk': 'slk',  # Slovak
    'bg': 'bul',  # Bulgarian
    'hr': 'hrv',  # Croatian
    'sr': 'srp',  # Serbian
    'sl': 'slv',  # Slovenian
    'lt': 'lit',  # Lithuanian
    'lv': 'lav',  # Latvian
    'et': 'est',  # Estonian
    'ca': 'cat',  # Catalan
    'eu': 'eus',  # Basque
    'gl': 'glg',  # Galician
    'af': 'afr',  # Afrikaans
    'is': 'isl',  # Icelandic
    'sq': 'sqi',  # Albanian
    'ka': 'kat',  # Georgian
    'hy': 'hye',  # Armenian
    'az': 'aze',  # Azerbaijani
    'uz': 'uzb',  # Uzbek
    'kk': 'kaz',  # Kazakh
    'mn': 'mon',  # Mongolian
    'ur': 'urd',  # Urdu
    'ne': 'nep',  # Nepali
    'si': 'sin',  # Sinhala
    'my': 'mya',  # Burmese
    'km': 'khm',  # Khmer
    'lo': 'lao',  # Lao
    'am': 'amh',  # Amharic
    'ti': 'tir',  # Tigrinya
    'yo': 'yor',  # Yoruba
    'ig': 'ibo',  # Igbo
    'zu': 'zul',  # Zulu
    'xh': 'xho',  # Xhosa
    'st': 'sot',  # Sotho
    'sn': 'sna',  # Shona
    'ha': 'hau',  # Hausa
    'mg': 'mlg',  # Malagasy
}

# Reverse mapping: ISO 639-3 to ISO 639-1
ISO_639_3_TO_1 = {v: k for k, v in ISO_639_1_TO_3.items()}


def iso1_to_iso3(code: str) -> str:
    """
    Convert ISO 639-1 (2-letter) code to ISO 639-3 (3-letter).
    
    Args:
        code: 2-letter language code (e.g., 'en', 'fr')
        
    Returns:
        3-letter language code (e.g., 'eng', 'fra')
        
    Raises:
        ValueError: If code is not recognized
    """
    code = code.lower()
    if code not in ISO_639_1_TO_3:
        raise ValueError(f"Unknown ISO 639-1 code: {code}")
    return ISO_639_1_TO_3[code]


def iso3_to_iso1(code: str) -> str:
    """
    Convert ISO 639-3 (3-letter) code to ISO 639-1 (2-letter).
    
    Args:
        code: 3-letter language code (e.g., 'eng', 'fra')
        
    Returns:
        2-letter language code (e.g., 'en', 'fr')
        
    Raises:
        ValueError: If code is not recognized or has no ISO 639-1 equivalent
    """
    code = code.lower()
    if code not in ISO_639_3_TO_1:
        raise ValueError(f"Unknown or no ISO 639-1 equivalent for: {code}")
    return ISO_639_3_TO_1[code]


def is_valid_iso1(code: str) -> bool:
    """Check if code is a valid ISO 639-1 code."""
    return code.lower() in ISO_639_1_TO_3


def is_valid_iso3(code: str) -> bool:
    """Check if code is a valid ISO 639-3 code."""
    return code.lower() in ISO_639_3_TO_1


def get_all_iso1_codes():
    """Get all supported ISO 639-1 codes."""
    return sorted(ISO_639_1_TO_3.keys())


def get_all_iso3_codes():
    """Get all supported ISO 639-3 codes."""
    return sorted(ISO_639_3_TO_1.keys())


if __name__ == "__main__":
    # Test conversions
    print("Language Code Conversion Test")
    print("=" * 50)
    
    test_cases = [
        ('en', 'eng'),
        ('fr', 'fra'),
        ('de', 'deu'),
        ('hi', 'hin'),
        ('ko', 'kor'),
        ('bn', 'ben'),
    ]
    
    print("\nISO 639-1 → ISO 639-3:")
    for iso1, expected_iso3 in test_cases:
        iso3 = iso1_to_iso3(iso1)
        status = "✓" if iso3 == expected_iso3 else "✗"
        print(f"  {status} {iso1} → {iso3}")
    
    print("\nISO 639-3 → ISO 639-1:")
    for expected_iso1, iso3 in test_cases:
        iso1 = iso3_to_iso1(iso3)
        status = "✓" if iso1 == expected_iso1 else "✗"
        print(f"  {status} {iso3} → {iso1}")
    
    print(f"\nTotal supported languages: {len(ISO_639_1_TO_3)}")
    print(f"ISO 639-1 codes: {len(get_all_iso1_codes())}")
    print(f"ISO 639-3 codes: {len(get_all_iso3_codes())}")