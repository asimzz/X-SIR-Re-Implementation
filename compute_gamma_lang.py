#!/usr/bin/env python3
"""
Compute γ_lang (language-specific green token fraction) from validation data.

For each pivot language, loads 500 human-written validation texts, runs KGW
detection, and computes the average green_fraction. This gives the empirical
γ_lang that corrects for tokenizer bias in low-resource languages.

Output: gamma_lang.json mapping pivot_lang -> γ_lang
"""

import os
import json
import argparse
import torch
import tqdm
import numpy as np

from transformers import AutoTokenizer
from src_watermark.kgw.extended_watermark_processor import WatermarkDetector as KGWDetector
from utils import read_jsonl


def compute_gamma_for_language(detector, val_data):
    """Compute average green_fraction for a list of validation texts."""
    green_fractions = []
    for item in val_data:
        text = item.get("response", "")
        if not text:
            continue
        try:
            result = detector.detect(text)
            gf = result.get("green_fraction")
            if gf is not None and gf == gf:  # not NaN
                green_fractions.append(float(gf))
        except (ValueError, RuntimeError):
            continue
    return green_fractions


def main():
    parser = argparse.ArgumentParser(description="Compute γ_lang from validation data")
    parser.add_argument("--base_model", type=str, required=True, help="Model name for tokenizer")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with mc4.{lang}.val.jsonl files")
    parser.add_argument("--z_score_dir", type=str, default=None,
                        help="Directory with mc4.{lang}.val.z_score.jsonl files (if different from input_dir)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--gamma", type=float, default=0.25, help="KGW gamma for green list")
    parser.add_argument("--seed", type=int, default=0, help="KGW seed")
    parser.add_argument("--seeding_scheme", type=str, default="minhash", help="KGW seeding scheme")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    detector = KGWDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seed=args.seed,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )

    # Find all validation files
    val_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.startswith("mc4.") and f.endswith(".val.jsonl")
    ])

    if not val_files:
        print(f"No validation files found in {args.input_dir}")
        return

    print(f"Found {len(val_files)} validation files")

    gamma_lang = {}
    with torch.no_grad():
        for val_file in tqdm.tqdm(val_files):
            # Extract language code: mc4.{lang}.val.jsonl -> lang
            lang = val_file.replace("mc4.", "").replace(".val.jsonl", "")

            val_data = read_jsonl(os.path.join(args.input_dir, val_file))
            green_fractions = compute_gamma_for_language(detector, val_data)

            if green_fractions:
                mean_gf = float(np.mean(green_fractions))
                std_gf = float(np.std(green_fractions))
                gamma_lang[lang] = {
                    "gamma_lang": mean_gf,
                    "std": std_gf,
                    "n_samples": len(green_fractions),
                }
                print(f"  {lang}: γ_lang={mean_gf:.4f} (std={std_gf:.4f}, n={len(green_fractions)})")
            else:
                print(f"  {lang}: no valid samples")

    # Save
    with open(args.output_file, "w") as f:
        json.dump(gamma_lang, f, indent=2)

    print(f"\nSaved γ_lang for {len(gamma_lang)} languages to {args.output_file}")

    # Print summary of extreme values
    if gamma_lang:
        sorted_langs = sorted(gamma_lang.items(), key=lambda x: x[1]["gamma_lang"])
        print("\nLowest γ_lang (most biased away from 0.25):")
        for lang, info in sorted_langs[:10]:
            print(f"  {lang}: {info['gamma_lang']:.4f}")
        print("\nHighest γ_lang:")
        for lang, info in sorted_langs[-5:]:
            print(f"  {lang}: {info['gamma_lang']:.4f}")


if __name__ == "__main__":
    main()
