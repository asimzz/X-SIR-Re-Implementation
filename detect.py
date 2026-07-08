import os
import tqdm
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from src_watermark.xsir.watermark import (
    WatermarkWindow as XSIRWindow,
    WatermarkContext as XSIRContext,
)
from src_watermark.kgw.extended_watermark_processor import (
    WatermarkDetector as KGWDetector
)
from src_watermark.uw.detect import Detector as UWDetector
from src_watermark.distortion_free.watermark import DistortionFreeDetector
from src_watermark.semstamp.detector import SemStampDetector

import numpy as np
from utils import read_jsonl, append_jsonl

def get_length(text, tokenizer):
    return len(tokenizer.encode(text))

def is_nan(nan):
    return nan != nan

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load watermark detector
    if args.watermark_method in ["xsir", "sir"]:
        if args.watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
            watermark_detector = XSIRWindow(
                device,
                args.window_size,
                tokenizer
            )
        elif args.watermark_type == "context":
            watermark_detector = XSIRContext(
                device,
                args.chunk_size,
                tokenizer,
                mapping_file=args.mapping_file,
                delta=args.delta,
                transform_model_path=args.transform_model,
                embedding_model=args.embedding_model
            )
        else:
            raise ValueError(f"Incorrect watermark type: {args.watermark_type}")
    elif args.watermark_method == "kgw":
        watermark_detector = KGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma, # should match original setting
            seed=args.seed, # should match original setting
            seeding_scheme=args.seeding_scheme, # should match original setting
            device=device, # must match the original rng device type
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
    elif args.watermark_method == "uw":
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", trust_remote_code=True)
        watermark_detector = UWDetector(
            model=model,
            tokenizer=tokenizer
        )
    elif args.watermark_method in ["its", "exp"]:
        # Distortion-free detector: returns {"z_score": -log(p_value)}. Exact permutation
        # test by default; pass --wm_fast with --wm_null_file for the precomputed-null path.
        null_results = None
        if args.wm_fast:
            if not args.wm_null_file:
                raise ValueError("--wm_fast requires --wm_null_file")
            null_results = np.load(args.wm_null_file)
        watermark_detector = DistortionFreeDetector(
            method=args.watermark_method,
            key=args.wm_key,
            n=args.wm_n,
            k=args.wm_k,
            gamma=args.wm_gamma,
            tokenizer=tokenizer,
            null_results=null_results,
            n_runs=args.wm_n_runs,
        )
    elif args.watermark_method == "semstamp":
        watermark_detector = SemStampDetector(
            embedder_name=args.embedding_model,
            lsh_dim=args.sp_dim,
            lmbd=args.lmbd,
            device=str(device),
        )
    else:
        raise ValueError(f"Incorrect watermark method: {args.watermark_method}")

    # Load data
    done_data = read_jsonl(args.output_file) if os.path.isfile(args.output_file) else []
    detect_data = read_jsonl(args.detect_file)
    if len(detect_data) == len(done_data):
        print("All data has been processed. Exiting...")
        return

    # Detect
    detect_data = detect_data[len(done_data):]
    with torch.no_grad():
        for dd in tqdm.tqdm(detect_data):
            try:
                detect_res = watermark_detector.detect(dd["response"])
            except ValueError as e:
                if "Must have at least" in str(e):
                    # Input is too short
                    detect_res = {"z_score": None}
                else:
                    raise e
            z_score = detect_res["z_score"]
            biases = detect_res["biases"] if "biases" in detect_res else None
            num_green_tokens = detect_res.get("num_green_tokens")
            num_tokens_scored = detect_res.get("num_tokens_scored")
            if is_nan(z_score):
                z_score = None
            output = {"z_score": z_score, "prompt": dd["prompt"], "response": dd["response"], "biases": biases}
            # Preserve per-example attack metadata (random-attack pipeline) so the
            # no-defense baseline can be broken down by attack-language tier.
            if "attack_lang" in dd:
                output["attack_lang"] = dd["attack_lang"]
                output["attack_tier"] = dd.get("attack_tier")
            if num_green_tokens is not None:
                output["num_green_tokens"] = int(num_green_tokens) if not is_nan(num_green_tokens) else None
            if num_tokens_scored is not None:
                output["num_tokens_scored"] = int(num_tokens_scored) if not is_nan(num_tokens_scored) else None
            append_jsonl(args.output_file, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare the z-scores of strings in detect_file.')
    # Model
    parser.add_argument('--base_model', type=str, required=True, help="Base model path. Only tokenizer is used.")

    # Data
    parser.add_argument('--detect_file', type=str, required=True, help="File to detect the z-scores.")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to write the z-scores.")

    # Watermark
    parser.add_argument('--watermark_method', type=str, choices=["xsir", "kgw", "sir", "uw", "its", "exp", "semstamp"], required=True, help="Watermarking method")
    parser.add_argument('--delta', type=float, default=None, help="bias of logit")
    parser.add_argument('--seed', type=int, default=0, help="Seed for watermarking")

    # X-SIR
    parser.add_argument('--watermark_type', type=str, default="context")
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--mapping_file', type=str, default="mapping.json")
    parser.add_argument('--transform_model', type=str, default="model/transform_model_x-sbert_test.pth")
    parser.add_argument('--embedding_model', type=str, default="paraphrase-multilingual-mpnet-base-v2")

    # KGW
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--seeding_scheme', type=str, default="minhash")

    # Distortion-free (ITS / EXP). key/n must match generation; k/gamma must match the null.
    parser.add_argument('--wm_key', type=int, default=42, help="Secret key/seed for ITS/EXP")
    parser.add_argument('--wm_n', type=int, default=256, help="Watermark sequence length for ITS/EXP")
    parser.add_argument('--wm_k', type=int, default=None,
                        help="Alignment block length. None=whole sequence (exact path only); "
                             "fixed int required for the fast path")
    parser.add_argument('--wm_gamma', type=float, default=1.0, help="Levenshtein indel cost for ITS/EXP")
    parser.add_argument('--wm_n_runs', type=int, default=100, help="Permutation runs (exact path)")
    parser.add_argument('--wm_fast', action="store_true", help="Use fast_permutation_test with a precomputed null")
    parser.add_argument('--wm_null_file', type=str, default=None, help="Path to {lang}.npy null distribution (fast path)")

    # SemStamp (uses --embedding_model as the sentence encoder).
    parser.add_argument('--sp_dim', type=int, default=3, help="LSH dimension (2^sp_dim regions)")
    parser.add_argument('--lmbd', type=float, default=0.25, help="Green-region acceptance rate")

    args = parser.parse_args()

    # Manually set default value for delta based on watermark_method
    if args.watermark_method == "kgw" and args.delta is None:
        args.delta = 2
    elif args.watermark_method in ["xsir", "sir"] and args.delta is None:
        args.delta = 1

    main(args)