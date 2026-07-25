#!/usr/bin/env python3
"""
Gemini Backtranslation Pipeline for STEAM BO Integration

Drop-in replacement for RealtimeBacktranslator that uses Gemini (via the
OpenAI-compatible Generative Language API) instead of Google Translate.

Exposes the same translate_text(text, source_lang, target_lang) interface the
STEAM BO detector relies on, so it can be swapped in without changing the
detection logic. Reuses the Gemini call pattern from attack/gemini_translate.py.
"""

import os
import time
import logging
from typing import Optional

from openai import OpenAI
from langcodes import Language

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2


class GeminiBacktranslator:
    """
    Real-time backtranslation using Gemini via the OpenAI-compatible client.

    Faithful (low-temperature) translation for detection-time back-translation:
    target_lang → pivot_lang.
    """

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 rate_limit_delay: float = 0.1,
                 max_retries: int = 3):
        """
        Initialize the Gemini backtranslator.

        Args:
            model: Gemini model id. Overridable via the GEMINI_MODEL env var.
            temperature: Sampling temperature. Overridable via GEMINI_TEMPERATURE.
            rate_limit_delay: Delay between API calls to avoid rate limiting.
            max_retries: Retries (with backoff) on transient API failures.
        """
        self.model = os.getenv("GEMINI_MODEL", model)
        env_temp = os.getenv("GEMINI_TEMPERATURE")
        self.temperature = float(env_temp) if env_temp is not None else temperature
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.translation_cache = {}  # Cache for repeated translations

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set the GEMINI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

        # Language code normalization for langcodes / Gemini compatibility.
        self.lang_code_map = {
            "iw": "he",     # Hebrew: legacy code -> current ISO code
        }

        self.logger.info(f"GeminiBacktranslator ready (model={self.model}, temperature={self.temperature})")

    def normalize_lang_code(self, lang_code: str) -> str:
        """Normalize language codes for langcodes/Gemini compatibility."""
        return self.lang_code_map.get(lang_code, lang_code)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text with Gemini.

        Args:
            text: Text to translate.
            source_lang: Source language code (ISO-1).
            target_lang: Target language code (ISO-1).

        Returns:
            Translated text, or None if translation failed.
        """
        source_lang = self.normalize_lang_code(source_lang)
        target_lang = self.normalize_lang_code(target_lang)

        cache_key = (text[:100], source_lang, target_lang)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        prompt = (
            f"Translate the following {Language.make(language=source_lang).display_name()} text "
            f"to {Language.make(language=target_lang).display_name()}:\n\n{text}"
        )

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful translator."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )

                translated_text = response.choices[0].message.content.strip()
                self.translation_cache[cache_key] = translated_text
                self.logger.debug(f"Translated {source_lang}→{target_lang}: {text[:50]}...")
                return translated_text

            except Exception as e:
                wait = self.rate_limit_delay * (2 ** attempt)
                self.logger.error(
                    f"Translation failed {source_lang}→{target_lang} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

        return None


def test_translation():
    """Test function for the Gemini translation pipeline."""
    translator = GeminiBacktranslator()

    test_text = "Bonjour le monde, ceci est un test."
    print("Testing Gemini translation pipeline (fr → de)...")
    translated = translator.translate_text(test_text, "fr", "de")
    print(f"Original French: {test_text}")
    print(f"Translated to German: {translated}")

    # Second identical call should hit the cache.
    cached = translator.translate_text(test_text, "fr", "de")
    print(f"Cache hit matches: {cached == translated}")


if __name__ == "__main__":
    test_translation()
