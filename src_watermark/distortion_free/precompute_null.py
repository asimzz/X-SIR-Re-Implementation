#!/usr/bin/env python3
"""Precompute per-language null distributions for the ITS/EXP fast detection path.

Human validation texts are (by definition) unwatermarked, so the test statistic evaluated on
them under the fixed key is a sample from the null. We collect those samples per language and
save them sorted; `fast_permutation_test` then turns a new statistic into a p-value via
`searchsorted`. This is the ITS/EXP analogue of the KGW per-language `gamma_lang.json` and is
what makes STEAM (which re-detects each text once per pivot back-translation) tractable.

Output: {output_dir}/{lang}.npy   (1-D float32, sorted ascending)

Example:
  python -m src_watermark.distortion_free.precompute_null \
      --base_model CohereForAI/aya-23-8B --method exp \
      --wm_key 42 --wm_n 256 --wm_k 40 --wm_gamma 1.0 \
      --val_dir data/dataset/mc4 --output_dir gen/aya-23-8B/exp_seed42/null \
      --langs fr de es it pt ru ja --num_texts 500
"""
import os
import argparse

import numpy as np
import tqdm
from transformers import AutoTokenizer

from src_watermark.distortion_free.watermark import DistortionFreeDetector
from utils import read_jsonl


def _val_path(val_dir, lang):
    return os.path.join(val_dir, f"mc4.{lang}.val.jsonl")


def build_null_for_language(detector, val_file, num_texts):
    data = read_jsonl(val_file)[:num_texts]
    stats = []
    for d in tqdm.tqdm(data, desc=os.path.basename(val_file), leave=False):
        text = d.get("response") or d.get("prompt") or ""
        if not text:
            continue
        try:
            stats.append(detector.compute_statistic(text))
        except ValueError:
            # too short for the chosen k -> not a usable null sample
            continue
    return np.sort(np.asarray(stats, dtype=np.float32))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    detector = DistortionFreeDetector(
        method=args.method,
        key=args.wm_key,
        n=args.wm_n,
        k=args.wm_k,          # fixed k (required for the fast path)
        gamma=args.wm_gamma,
        tokenizer=tokenizer,
        null_results=None,    # we only call compute_statistic here
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for lang in args.langs:
        val_file = _val_path(args.val_dir, lang)
        if not os.path.exists(val_file):
            print(f"⚠️  Missing validation file {val_file}; skipping {lang}")
            continue
        out_file = os.path.join(args.output_dir, f"{lang}.npy")
        if os.path.exists(out_file) and not args.overwrite:
            print(f"✓ {lang}: null already exists ({out_file}); use --overwrite to rebuild")
            continue
        null = build_null_for_language(detector, val_file, args.num_texts)
        if null.size == 0:
            print(f"⚠️  No usable null samples for {lang} (all too short?); skipping")
            continue
        np.save(out_file, null)
        print(f"✓ {lang}: saved {null.size} null samples -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute per-language null distributions (ITS/EXP)")
    parser.add_argument("--base_model", type=str, required=True, help="Model id (tokenizer only)")
    parser.add_argument("--method", type=str, choices=["its", "exp"], required=True)
    parser.add_argument("--wm_key", type=int, default=42)
    parser.add_argument("--wm_n", type=int, default=256)
    parser.add_argument("--wm_k", type=int, required=True,
                        help="Fixed block length; MUST match the detection --wm_k")
    parser.add_argument("--wm_gamma", type=float, default=1.0)
    parser.add_argument("--val_dir", type=str, default="data/dataset/mc4",
                        help="Directory holding mc4.{lang}.val.jsonl human negatives")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write {lang}.npy null distributions")
    parser.add_argument("--langs", type=str, nargs="+", required=True,
                        help="Languages (ISO-639-1) to build nulls for")
    parser.add_argument("--num_texts", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")

    main(parser.parse_args())
