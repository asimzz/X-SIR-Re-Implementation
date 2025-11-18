import os
import tqdm
import torch
import argparse
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM
from src_watermark.xsir.watermark import (
    WatermarkWindow as XSIRWindow,
    WatermarkContext as XSIRContext,
)
from src_watermark.kgw.extended_watermark_processor import (
    WatermarkDetector as KGWDetector
)
from src_watermark.uw.detect import Detector as UWDetector

from utils import read_jsonl, append_jsonl

def get_length(text, tokenizer):
    return len(tokenizer.encode(text))

def is_nan(nan):
    return nan != nan

def get_text_bins(text, tokenizer):
    """
    Divide text into different length bins by chunking:
    - short: first 50 tokens
    - medium: first 150 tokens
    - long: full text (if > 150 tokens)
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    bins = {}

    # Short: first 50 tokens
    if len(tokens) >= 50:
        short_tokens = tokens[:50]
        bins['short'] = tokenizer.decode(short_tokens)

    # Medium: first 150 tokens
    if len(tokens) >= 150:
        medium_tokens = tokens[:150]
        bins['medium'] = tokenizer.decode(medium_tokens)

    # Long: all tokens (if > 150)
    if len(tokens) > 150:
        bins['long'] = text

    return bins

def get_binned_output_files(output_file):
    """Generate output filenames for each bin"""
    base_name = output_file.replace('.jsonl', '')
    return {
        'short': f"{base_name}.short.z_score.jsonl",
        'medium': f"{base_name}.medium.z_score.jsonl",
        'long': f"{base_name}.long.z_score.jsonl"
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load watermark detector (same logic as detect.py)
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
    else:
        raise ValueError(f"Incorrect watermark method: {args.watermark_method}")

    # Get binned output files
    bin_output_files = get_binned_output_files(args.output_file)

    # Load existing data for each bin
    done_data_bins = {}
    for bin_name, bin_file in bin_output_files.items():
        done_data_bins[bin_name] = read_jsonl(bin_file) if os.path.isfile(bin_file) else []

    # Load input data
    detect_data = read_jsonl(args.detect_file)

    print(f"📊 Processing {len(detect_data)} texts with binned analysis")
    print(f"📦 Bins: Short (<50 tokens), Medium (50-150 tokens), Long (>150 tokens)")

    # Track statistics
    bin_counts = defaultdict(int)
    processed_count = 0

    # Detect with binning
    with torch.no_grad():
        for dd in tqdm.tqdm(detect_data, desc="Detecting with binned analysis"):
            # Get text bins by chunking the response
            text_bins = get_text_bins(dd["response"], tokenizer)

            # Process each available bin for this text
            for bin_name, bin_text in text_bins.items():
                bin_counts[bin_name] += 1

                # Skip if already processed in this bin
                if len(done_data_bins[bin_name]) >= bin_counts[bin_name]:
                    continue

                try:
                    # Detect watermark on the binned text chunk
                    detect_res = watermark_detector.detect(bin_text)
                    z_score = detect_res["z_score"]
                    biases = detect_res["biases"] if "biases" in detect_res else None

                    if is_nan(z_score):
                        z_score = None

                    # Prepare output data
                    output_data = {
                        "z_score": z_score,
                        "prompt": dd["prompt"],
                        "response": bin_text,  # Store the binned text chunk
                        "original_response": dd["response"],  # Keep original for reference
                        "biases": biases,
                        "bin_length": get_length(bin_text, tokenizer),
                        "original_length": get_length(dd["response"], tokenizer),
                        "length_bin": bin_name
                    }

                    # Append to appropriate bin file
                    append_jsonl(bin_output_files[bin_name], output_data)
                    processed_count += 1

                except ValueError as e:
                    if "Must have at least" in str(e):
                        # Binned text is too short for watermark detection
                        output_data = {
                            "z_score": None,
                            "prompt": dd["prompt"],
                            "response": bin_text,
                            "original_response": dd["response"],
                            "biases": None,
                            "bin_length": get_length(bin_text, tokenizer),
                            "original_length": get_length(dd["response"], tokenizer),
                            "length_bin": bin_name
                        }
                        append_jsonl(bin_output_files[bin_name], output_data)
                        processed_count += 1
                    else:
                        raise e

    # Print summary statistics
    print(f"\n📈 Binned Analysis Summary:")
    print(f"   Total processed: {processed_count}")
    for bin_name in ['short', 'medium', 'long']:
        count = bin_counts[bin_name]
        print(f"   {bin_name.capitalize()} texts: {count} ({count/sum(bin_counts.values())*100:.1f}%)")
        print(f"   -> Saved to: {bin_output_files[bin_name]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect z-scores with length-based binning analysis.')

    # Model
    parser.add_argument('--base_model', type=str, required=True, help="Base model path. Only tokenizer is used.")

    # Data
    parser.add_argument('--detect_file', type=str, required=True, help="File to detect the z-scores.")
    parser.add_argument('--output_file', type=str, required=True, help="Base output file name (will create .short/.medium/.long variants)")

    # Watermark
    parser.add_argument('--watermark_method', type=str, choices=["xsir", "kgw", "sir", "uw"], required=True, help="Watermarking method")
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

    args = parser.parse_args()

    # Manually set default value for delta based on watermark_method
    if args.watermark_method == "kgw" and args.delta is None:
        args.delta = 2
    elif args.watermark_method in ["xsir", "sir"] and args.delta is None:
        args.delta = 1

    main(args)