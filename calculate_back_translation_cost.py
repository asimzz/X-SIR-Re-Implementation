#!/usr/bin/env python3

import os
import json
import argparse
import re
from transformers import AutoTokenizer
from utils import read_jsonl

def get_token_count(text, tokenizer):
    """Count tokens in text using the given tokenizer."""
    return len(tokenizer.encode(text))

def extract_target_lang_from_filename(filename):
    """Extract target language from filename like 'mc4.en-ar.mod.jsonl' -> 'ar'"""
    match = re.search(r'mc4\.en-([^.]+)\.mod\.jsonl', filename)
    if match:
        return match.group(1)
    return None

def find_back_translation_files(base_dir, target_lang, org_langs):
    """Find all back-translation files for the given target language."""
    back_files = []
    for org_lang in org_langs:
        if org_lang != target_lang:
            back_file = os.path.join(base_dir, f"mc4.{target_lang}-{org_lang}-back.mod.jsonl")
            if os.path.exists(back_file):
                back_files.append((org_lang, back_file))
    return back_files

def calculate_file_tokens(file_path, tokenizer):
    """Calculate total tokens in a JSONL file."""
    try:
        data = read_jsonl(file_path)
        total_tokens = 0
        for entry in data:
            if "response" in entry:
                total_tokens += get_token_count(entry["response"], tokenizer)
        return total_tokens, len(data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0

def main(args):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Extract target language from input file
    input_filename = os.path.basename(args.input_file)
    target_lang = extract_target_lang_from_filename(input_filename)

    if not target_lang:
        print(f"❌ Could not extract target language from filename: {input_filename}")
        print("Expected format: mc4.en-{lang}.mod.jsonl")
        return

    print(f"📊 Calculating back-translation costs for target language: {target_lang}")

    # Get base directory
    base_dir = os.path.dirname(args.input_file)

    # Define original languages - updated to include all 17 languages
    org_langs = ["en", "bn", "de", "es", "fa", "fr", "hi", "it", "iw", "ja", "ko", "nl", "pl", "pt", "ru", "ta", "uk", "vi"]

    # Calculate input tokens
    print(f"📥 Counting input tokens in: {args.input_file}")
    input_tokens, input_samples = calculate_file_tokens(args.input_file, tokenizer)
    print(f"   Input: {input_tokens:,} tokens across {input_samples} samples")

    # Find and process back-translation files
    back_files = find_back_translation_files(base_dir, target_lang, org_langs)

    if not back_files:
        print(f"❌ No back-translation files found for {target_lang}")
        return

    print(f"\n📤 Counting output tokens in back-translation files:")
    total_output_tokens = 0
    total_output_samples = 0

    for org_lang, back_file in back_files:
        output_tokens, output_samples = calculate_file_tokens(back_file, tokenizer)
        total_output_tokens += output_tokens
        total_output_samples += output_samples
        print(f"   {target_lang} → {org_lang}: {output_tokens:,} tokens ({output_samples} samples)")

    # Calculate aggregated costs
    print(f"\n💰 Cost Summary:")
    print(f"   Target Language: {target_lang}")
    print(f"   Input Tokens:    {input_tokens:,}")
    print(f"   Output Tokens:   {total_output_tokens:,}")
    print(f"   Total Tokens:    {input_tokens + total_output_tokens:,}")
    print(f"   Back-translation Files: {len(back_files)}")
    print(f"   Cost Multiplier: {total_output_tokens / input_tokens:.2f}x" if input_tokens > 0 else "   Cost Multiplier: N/A")

    # Save results if output file specified
    if args.output_file:
        results = {
            "target_language": target_lang,
            "input_file": args.input_file,
            "input_tokens": input_tokens,
            "input_samples": input_samples,
            "output_tokens": total_output_tokens,
            "output_samples": total_output_samples,
            "total_tokens": input_tokens + total_output_tokens,
            "cost_multiplier": total_output_tokens / input_tokens if input_tokens > 0 else 0,
            "back_translation_files": len(back_files),
            "breakdown": [
                {
                    "source_lang": target_lang,
                    "target_lang": org_lang,
                    "file": back_file,
                    "tokens": calculate_file_tokens(back_file, tokenizer)[0],
                    "samples": calculate_file_tokens(back_file, tokenizer)[1]
                }
                for org_lang, back_file in back_files
            ]
        }

        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate back-translation costs for watermark analysis"
    )

    # Required arguments
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file (e.g., mc4.en-ar.mod.jsonl)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model for tokenizer (e.g., meta-llama/Llama-3.2-1B)"
    )

    # Optional arguments
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output JSON file to save results"
    )

    args = parser.parse_args()
    main(args)