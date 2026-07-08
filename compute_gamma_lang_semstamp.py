#!/usr/bin/env python3
"""Compute per-language γ_lang for SemStamp (sentence-level green fraction).

Analogue of compute_gamma_lang.py but at the SENTENCE level and using the multilingual
SemStamp detector. For each language, run SemStamp detection over human validation texts and
average the per-text green-sentence fraction. The resulting gamma_lang.json is consumed by the
STEAM scoring path (steam_bo_detector.py --watermark_method semstamp) exactly like the KGW one.

Output: gamma_lang.json mapping lang -> {gamma_lang, std, n_samples}
"""
import os
import json
import argparse

import numpy as np
import tqdm

from src_watermark.semstamp.detector import SemStampDetector
from utils import read_jsonl


def compute_gamma_for_language(detector, val_data):
    fracs = []
    for item in val_data:
        text = item.get("response", "") or item.get("prompt", "")
        if not text:
            continue
        try:
            result = detector.detect(text)
            gf = result.get("green_fraction")
            if gf is not None and gf == gf:  # not NaN
                fracs.append(float(gf))
        except (ValueError, RuntimeError):
            continue
    return fracs


def main():
    parser = argparse.ArgumentParser(description="Compute SemStamp sentence-level γ_lang")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with mc4.{lang}.val.jsonl files")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--embedding_model", type=str, default="paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--sp_dim", type=int, default=3)
    parser.add_argument("--lmbd", type=float, default=0.25)
    parser.add_argument("--num_texts", type=int, default=500)
    parser.add_argument("--langs", type=str, nargs="*", default=None,
                        help="Restrict to these languages (default: all mc4.*.val.jsonl found)")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = SemStampDetector(
        embedder_name=args.embedding_model, lsh_dim=args.sp_dim, lmbd=args.lmbd, device=device)

    if args.langs:
        val_files = [f"mc4.{l}.val.jsonl" for l in args.langs]
    else:
        val_files = sorted(f for f in os.listdir(args.input_dir)
                           if f.startswith("mc4.") and f.endswith(".val.jsonl"))
    if not val_files:
        print(f"No validation files found in {args.input_dir}")
        return

    gamma_lang = {}
    for val_file in tqdm.tqdm(val_files):
        path = os.path.join(args.input_dir, val_file)
        if not os.path.exists(path):
            print(f"  skip missing {val_file}")
            continue
        lang = val_file.replace("mc4.", "").replace(".val.jsonl", "")
        fracs = compute_gamma_for_language(detector, read_jsonl(path)[:args.num_texts])
        if fracs:
            gamma_lang[lang] = {
                "gamma_lang": float(np.mean(fracs)),
                "std": float(np.std(fracs)),
                "n_samples": len(fracs),
            }
            print(f"  {lang}: γ_lang={gamma_lang[lang]['gamma_lang']:.4f} (n={len(fracs)})")
        else:
            print(f"  {lang}: no valid samples")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(gamma_lang, f, indent=2)
    print(f"\nSaved γ_lang for {len(gamma_lang)} languages to {args.output_file}")


if __name__ == "__main__":
    main()
