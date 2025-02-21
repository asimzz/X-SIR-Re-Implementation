import os
import json
import random
import argparse
import networkx as nx
from transformers import AutoTokenizer

# Set seed for reproducibility
random.seed(0)

# Maximum size of connected components
CC_MAX_SIZE = 250

def main():
    parser = argparse.ArgumentParser(description="Generate mappings.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dictionary", type=str, required=True, help="Dictionary path. One line per entry.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = tokenizer.get_vocab()

    print(f"Vocabulary size (from get_vocab): {len(vocab)}")
    print(f"Vocabulary size (from tokenizer): {tokenizer.vocab_size}")

    # Ensure only valid tokens exist
    valid_token_ids = set(tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size)))

    # Load dictionary as edges
    edges = []
    with open(args.dictionary) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 2:
                src_token, tgt_token = tokens
                if src_token in valid_token_ids and tgt_token in valid_token_ids:
                    edges.append((src_token, tgt_token))

    # Add self-loop for each valid token
    for token in valid_token_ids:
        edges.append((token, token))

    # Build graph & find connected components
    graph = nx.Graph(edges)
    connected_components_node = list(nx.connected_components(graph))  # List of node sets
    connected_components_node.sort(key=len, reverse=True)  # Sort by size
    connected_components_graph = [graph.subgraph(ccn) for ccn in connected_components_node]  # List of graphs

    # Split connected components into clusters
    clusters = []
    for ccg in connected_components_graph:
        if len(ccg) <= CC_MAX_SIZE:
            clusters.append(list(ccg))
        else:
            resolution = 1 if len(ccg) <= 10000 else 10  # Higher resolution for large clusters
            cs = nx.community.louvain_communities(ccg, seed=0, resolution=resolution)
            clusters.extend(cs)
            print(f"Splitted {len(ccg)} nodes into {len(cs)} clusters")

    # Filter only valid tokens
    valid_clusters = []
    for c in clusters:
        valid_c = {token for token in c if token in valid_token_ids}
        if valid_c:
            valid_clusters.append(list(valid_c))

    # Ensure correct token count
    all_valid_tokens = [token for c in valid_clusters for token in c]

    if len(all_valid_tokens) != tokenizer.vocab_size:
        print(f"⚠️ Warning: Mismatch detected! all_valid_tokens={len(all_valid_tokens)}, tokenizer.vocab_size={tokenizer.vocab_size}")
        
        # Ensure the count does not exceed vocab size
        if len(all_valid_tokens) > tokenizer.vocab_size:
            all_valid_tokens = all_valid_tokens[:tokenizer.vocab_size]
        
        print(f"✅ Adjusted valid tokens count: {len(all_valid_tokens)}")

    # Generate cluster mappings
    cluster_mapping = [random.randint(0, 300 - 1) for _ in range(len(valid_clusters))]
    mapping = [None] * tokenizer.vocab_size

    for i, c in enumerate(valid_clusters):
        for token in c:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and mapping[token_id] is None:
                mapping[token_id] = cluster_mapping[i]

    # Ensure all values are assigned
    assert all(x is not None for x in mapping), "Error: Some tokens were not mapped properly."

    # Save mappings
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(mapping, f, indent=4)

    # Save clusters
    output_cluster_file = args.output_file.replace(".json", "_clusters.json")
    with open(output_cluster_file, "w") as f:
        valid_clusters.sort(key=len, reverse=True)  # Sort by size
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)

    # Print statistics
    print(f"✅ Vocabulary size: {tokenizer.vocab_size}")
    print(f"✅ Number of clusters: {len(valid_clusters)}")
    print(f"✅ Number of clusters with size ≥ 2: {len([cc for cc in valid_clusters if len(cc) >= 2])}")
    print(f"✅ Number of tokens in clusters with size ≥ 2: {sum(len(cc) for cc in valid_clusters if len(cc) >= 2)}")
    print(f"✅ Vocab coverage: {sum(len(cc) for cc in valid_clusters if len(cc) >= 2) / tokenizer.vocab_size * 100:.2f}%")

if __name__ == "__main__":
    main()
