#!/usr/bin/env python3
"""
Random per-example translation attack.

Instead of translating every example into a single fixed attack language (which
requires running all attack languages separately per source language), this
picks ONE attack language *per example*, uniformly at random from a curated
40-language pool, and translates the source-language text into it.

For a source language S this produces a single "mixed" file where each line was
attacked into a different, randomly-chosen language A (A != S). The sampled
language and its resource tier are recorded on each line as `attack_lang` /
`attack_tier` so downstream detection (steam_bo_detector.py --per_example_attack)
knows the back-translation direction and evaluation can break results down by tier.

Design notes:
  * Per-index seeding (RandomState(random_state + i)) makes the choice for line i
    depend only on i, so the run is fully reproducible AND resumable (interrupted
    runs resume from the last complete line and re-derive identical choices).
  * Run this twice with the SAME --random_state on the watermarked (.mod) and
    human (.hum) files so index i gets the same attack language in both, keeping
    the positive/negative sets comparable.
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from deep_translator import GoogleTranslator


# Curated 40-language attack pool (superset of the paper's 17), grouped by
# resource tier. Codes are Google-Translate / ISO-639-1 codes (zh-CN, iw as used
# elsewhere in this repo).
TIERS = {
    "high": ["fr", "de", "it", "es", "pt", "ar", "tr", "zh-CN", "id", "cs", "el", "ro"],
    "medium": ["pl", "nl", "ru", "hi", "ko", "ja", "th", "sr", "bg", "ms", "ur", "te", "ka", "hu"],
    "low": ["bn", "fa", "vi", "iw", "uk", "ta", "sw", "am", "hy", "my", "ne", "si", "km", "yo"],
}
DEFAULT_ATTACK_LANGS = TIERS["high"] + TIERS["medium"] + TIERS["low"]
LANG_TO_TIER = {lang: tier for tier, langs in TIERS.items() for lang in langs}


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize_for_google(code):
    """Map repo language labels to codes the deep_translator Google backend accepts.

    Note: deep_translator uses 'iw' (not 'he') for Hebrew and 'zh-CN' for Chinese,
    so 'iw' is left untouched here.
    """
    if code == "zh":
        return "zh-CN"
    return code


def main(args):
    src_lang = args.src_lang
    translation_part = args.translation_part
    attack_langs = [c.strip() for c in args.attack_langs.split(",") if c.strip()]

    # Exclude the source language (compared under Google normalization so that
    # e.g. zh vs zh-CN are treated as the same language).
    src_norm = normalize_for_google(src_lang)
    pool = [a for a in attack_langs if normalize_for_google(a) != src_norm]
    if not pool:
        raise ValueError(f"Attack pool is empty after excluding source '{src_lang}'.")

    input_data = read_jsonl(args.input_file)

    # Resume: keep already-translated lines (append mode).
    translated_data = read_jsonl(args.output_file) if os.path.exists(args.output_file) else []
    total = len(input_data)
    done = len(translated_data)
    print(f"Random-attack translating '{translation_part}' from src={src_lang}; "
          f"pool={len(pool)} langs; seed={args.random_state}")
    print(f"{total} samples found. {done} already translated.")
    if total == done:
        print("✅ Translation already completed. Skipping.")
        return

    # Cache one translator per (src, tgt) pair.
    translators = {}

    def get_translator(tgt_norm):
        if tgt_norm not in translators:
            translators[tgt_norm] = GoogleTranslator(source=src_norm, target=tgt_norm)
        return translators[tgt_norm]

    with open(args.output_file, "a", encoding="utf-8") as output_file:
        for idx in tqdm(range(done, total), desc="Translating", unit="line"):
            data = dict(input_data[idx])

            # Reproducible per-index choice of attack language.
            rng = np.random.RandomState(args.random_state + idx)
            attack_lang = str(rng.choice(pool))
            data["attack_lang"] = attack_lang
            data["attack_tier"] = LANG_TO_TIER.get(attack_lang)

            if translation_part in data and data[translation_part]:
                try:
                    tgt_norm = normalize_for_google(attack_lang)
                    translation = get_translator(tgt_norm).translate(data[translation_part])
                    if translation:
                        data[translation_part] = translation
                    # else: keep original text (translation returned empty)
                except Exception as e:
                    print(f"⚠️ Error on index {idx} (src={src_lang}->{attack_lang}): {e}")
                    # keep original text on failure
            output_file.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ Random-attack file saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random per-example translation attack.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--src_lang", type=str, required=True,
                        help="Source (generated) language code, e.g. fr")
    parser.add_argument("--translation_part", type=str, default="response",
                        help="Which field to translate (default: response)")
    parser.add_argument("--attack_langs", type=str, default=",".join(DEFAULT_ATTACK_LANGS),
                        help="Comma-separated attack-language pool (default: curated 40)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Base seed; line i uses RandomState(random_state + i)")
    args = parser.parse_args()
    main(args)
