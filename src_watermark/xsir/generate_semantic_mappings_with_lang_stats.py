import os
import re
import json
import random
import argparse
import networkx as nx
from collections import defaultdict
from transformers import AutoTokenizer

# Maximum size of connected components (used to decide clustering granularity)
CC_MAX_SIZE = 250


def parse_langs_from_filename(path):
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


def load_dictionary_files(dict_files):
    """
    Load multiple dictionary files and track which tokens belong to which languages.
    Returns:
        - edges: list of (src_token, tgt_token) tuples
        - token_languages: dict mapping token -> set of language codes
    """
    edges = []
    token_languages = defaultdict(set)

    for dict_file in dict_files:
        lang1, lang2 = parse_langs_from_filename(dict_file)

        with open(dict_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                src_token, tgt_token = parts[0], parts[1]
                edges.append((src_token, tgt_token))

                # Track language for each token
                token_languages[src_token].add(lang1)
                token_languages[tgt_token].add(lang2)

    return edges, token_languages


def compute_per_language_coverage(valid_clusters, token_languages, vocab):
    """
    Compute vocabulary coverage per language.
    Returns a dict: {lang: {total_tokens, tokens_in_clusters_size_2+, coverage_percent}}
    """
    # Get all tokens per language that are in vocab
    lang_tokens = defaultdict(set)
    for token, langs in token_languages.items():
        if token in vocab:
            for lang in langs:
                lang_tokens[lang].add(token)

    # Count how many tokens per language are in clusters with size >= 2
    lang_in_clusters = defaultdict(set)
    for cluster in valid_clusters:
        if len(cluster) >= 2:
            for token in cluster:
                if token in token_languages:
                    for lang in token_languages[token]:
                        lang_in_clusters[lang].add(token)

    # Compute coverage
    lang_stats = {}
    vocab_size = len(vocab)

    for lang in sorted(lang_tokens.keys()):
        total = len(lang_tokens[lang])
        in_clusters = len(lang_in_clusters[lang])
        coverage = (in_clusters / total * 100) if total > 0 else 0.0
        vocab_contribution = (in_clusters / vocab_size * 100) if vocab_size > 0 else 0.0

        lang_stats[lang] = {
            "total_tokens_in_vocab": total,
            "tokens_in_clusters_size_2+": in_clusters,
            "coverage_percent": round(coverage, 2),
            "vocab_contribution_percent": round(vocab_contribution, 2)
        }

    return lang_stats


def main():
    parser = argparse.ArgumentParser(description='Generate token-to-cluster mappings with per-language stats.')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--dictionary_files', type=str, nargs='+', required=True,
                        help='Multiple dictionary files (e.g., en-de.txt, en-fr.txt)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save token->cluster mapping JSON')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = tokenizer.get_vocab()

    # Load edges from dictionary files and track token languages
    edges, token_languages = load_dictionary_files(args.dictionary_files)

    # Add self-loop for all vocab tokens
    for token in vocab:
        edges.append((token, token))

    graph = nx.Graph(edges)
    connected_components_node = list(nx.connected_components(graph))
    connected_components_node.sort(key=len, reverse=True)
    connected_components_graph = [graph.subgraph(ccn) for ccn in connected_components_node]

    clusters = []
    for ccg in connected_components_graph:
        if len(ccg) <= CC_MAX_SIZE:
            clusters.append(list(ccg))
        else:
            resolution = 1 if len(ccg) <= 10000 else 10
            cs = nx.community.louvain_communities(ccg, seed=args.seed, resolution=resolution)
            clusters.extend(cs)
            print(f"[Seed {args.seed}] Split {len(ccg)} nodes into {len(cs)} clusters")

    # Filter clusters to keep only tokens in vocab
    valid_clusters = []
    for c in clusters:
        valid_c = [token for token in c if token in vocab]
        if valid_c:
            valid_clusters.append(valid_c)

    # Flatten clusters to check vocabulary coverage
    all_valid_tokens = [token for cluster in valid_clusters for token in cluster]
    assert len(all_valid_tokens) == len(vocab), \
        f"Vocabulary mismatch: {len(all_valid_tokens)} tokens mapped vs {len(vocab)} in vocab"

    # Compute per-language coverage
    lang_stats = compute_per_language_coverage(valid_clusters, token_languages, vocab)

    # Generate mapping
    cluster_ids = [random.randint(0, 299) for _ in range(len(valid_clusters))]
    mapping = [None] * len(vocab)
    for i, cluster in enumerate(valid_clusters):
        for token in cluster:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert mapping[token_id] is None
            mapping[token_id] = cluster_ids[i]

    assert all(x is not None for x in mapping), "Incomplete token-to-cluster mapping"

    # Save token-to-cluster mapping
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(mapping, f, indent=4)

    # Save clusters
    cluster_path = args.output_file.replace(".json", "_clusters.json")
    valid_clusters.sort(key=len, reverse=True)
    with open(cluster_path, "w") as f:
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)

    # Save per-language stats
    stats_path = args.output_file.replace(".json", "_lang_stats.json")
    with open(stats_path, "w") as f:
        json.dump(lang_stats, f, indent=4, ensure_ascii=False)

    # Print stats
    print(f"\n[Seed {args.seed}] Vocabulary size: {len(vocab)}")
    print(f"[Seed {args.seed}] Number of clusters: {len(valid_clusters)}")
    print(f"[Seed {args.seed}] Clusters with ≥2 tokens: {sum(len(c) >= 2 for c in valid_clusters)}")
    print(f"[Seed {args.seed}] Overall vocab coverage (%): {sum(len(c) for c in valid_clusters if len(c) >= 2) / len(vocab) * 100:.2f}")
    print(f"[Seed {args.seed}] Top 5 largest clusters: {[len(c) for c in sorted(valid_clusters, key=len, reverse=True)[:5]]}")

    # Print per-language coverage
    print(f"\n{'Language':<10} | {'Total Tokens':>12} | {'In Clusters':>12} | {'Lang Coverage %':>16} | {'Vocab Contrib %':>16}")
    print("-" * 80)
    for lang, stats in sorted(lang_stats.items()):
        print(f"{lang:<10} | {stats['total_tokens_in_vocab']:>12} | {stats['tokens_in_clusters_size_2+']:>12} | {stats['coverage_percent']:>16.2f} | {stats['vocab_contribution_percent']:>16.2f}")

    # Calculate total contribution
    total_contrib = sum(stats['vocab_contribution_percent'] for stats in lang_stats.values())
    print("-" * 80)
    print(f"{'TOTAL':<10} | {'':>12} | {'':>12} | {'':>16} | {total_contrib:>16.2f}")
    print(f"\nNote: 'Lang Coverage %' = percentage of that language's dictionary tokens in clusters")
    print(f"      'Vocab Contrib %' = percentage of total vocabulary contributed by that language")
    print(f"      The sum of 'Vocab Contrib %' shows the decomposition of the overall {sum(len(c) for c in valid_clusters if len(c) >= 2) / len(vocab) * 100:.2f}% coverage")

    print(f"\nSaved language stats to {stats_path}")

if __name__ == '__main__':
    main()
