import os
import re
import json
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Set

from transformers import AutoTokenizer
from utils import transform
from tqdm import tqdm


def parse_langs_from_filename(path: str) -> Tuple[str, str]:
    """
    Expect filenames like 'en-de.txt', 'fr_en.txt', etc.
    Returns (lang1, lang2). If not parsable, returns ('unk', 'unk').
    """
    name = os.path.basename(path)
    # normalize separators to '-'
    name_no_ext = re.sub(r"\.txt$", "", name)
    name_no_ext = name_no_ext.replace("_", "-")
    m = re.match(r"^([a-z]{2,3})-([a-z]{2,3})$", name_no_ext)
    if m:
        return m.group(1), m.group(2)
    return "unk", "unk"


def load_pairs(dict_file: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(dict_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # keep only first two columns (some files may have count/score)
            pairs.append((parts[0], parts[1]))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Count how many dictionary words exist as single tokens in a model tokenizer, per language."
    )
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--dicts", type=str, nargs="+", required=True, help="List of bi-dictionary files")
    parser.add_argument(
        "--append-meta-symbols",
        action="store_true",
        help="Pass True to utils.transform(..., append_meta_symbols=True)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="JSON file to write per-language counts/coverage",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab: Dict[str, int] = tokenizer.get_vocab()
    vocab_tokens: Set[str] = set(vocab.keys())

    # Per-language accounting
    # We’ll aggregate *unique words* seen and *unique hits* (words that map to a single vocab token)
    words_seen: Dict[str, Set[str]] = defaultdict(set)
    words_hit: Dict[str, Set[str]] = defaultdict(set)

    # Also keep raw counts per dictionary (optional, useful sanity)
    per_dict_stats = {}

    for dict_path in args.dicts:
        lang1, lang2 = parse_langs_from_filename(dict_path)
        pairs = load_pairs(dict_path)

        # Track stats for this dictionary
        d_stats = {
            "file": dict_path,
            "langs": [lang1, lang2],
            "total_pairs": 0,
            f"{lang1}_words_seen": 0,
            f"{lang2}_words_seen": 0,
            f"{lang1}_hits": 0,
            f"{lang2}_hits": 0,
        }

        for src, tgt in tqdm(pairs, desc=f"Scanning {os.path.basename(dict_path)}"):
            d_stats["total_pairs"] += 1

            # For each side, transform -> token candidates
            # We consider a "word exists in tokenizer" iff *any* transformed token equals a vocab token.
            # (Most SentencePiece/BPE setups need transform to add leading markers, etc.)
            for word, lang in ((src, lang1), (tgt, lang2)):
                if not word:
                    continue
                words_seen[lang].add(word)

                toks = transform(word, args.append_meta_symbols)
                # Is this word representable by a *single* tokenizer token?
                # i.e., transform returns exactly one token AND it exists in vocab.
                if word in vocab_tokens:
                    words_hit[lang].add(word)
                    if f"{lang}_hits" in d_stats:
                        d_stats[f"{lang}_hits"] += 1

        d_stats[f"{lang1}_words_seen"] = len(words_seen[lang1])
        d_stats[f"{lang2}_words_seen"] = len(words_seen[lang2])
        per_dict_stats[os.path.basename(dict_path)] = d_stats

    # Build per-language summary
    summary = {}
    lang_codes = sorted(set(list(words_seen.keys()) + list(words_hit.keys())))
    for lang in lang_codes:
        total = len(words_seen[lang])
        hits = len(words_hit[lang])
        coverage = (hits / total) * 100 if total > 0 else 0.0
        summary[lang] = {
            "unique_words_seen": total,
            "unique_words_hit_single_token": hits,
            "coverage_percent": round(coverage, 2),
        }

    # Write JSON report
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "append_meta_symbols": bool(args.append_meta_symbols),
                "per_language": summary,
                "per_dictionary": per_dict_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Pretty print table to stdout
    print("\n=== Per-language coverage (single-token hits) ===")
    print(f"{'Lang':<6} | {'Seen':>8} | {'Hits':>8} | {'Coverage %':>10}")
    print("-" * 40)
    for lang in lang_codes:
        row = summary[lang]
        print(f"{lang:<6} | {row['unique_words_seen']:>8} | {row['unique_words_hit_single_token']:>8} | {row['coverage_percent']:>10.2f}")
    print(f"\nWrote JSON report → {args.output_file}")
    

if __name__ == "__main__":
    main()
