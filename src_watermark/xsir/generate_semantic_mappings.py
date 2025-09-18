import os
import json
import heapq
import random
import argparse
import networkx as nx
from transformers import AutoTokenizer

# Maximum size of connected components (used to decide clustering granularity)
CC_MAX_SIZE = 250

def louvain_or_trivial(ccg, seed=0):
    """Return list[list[token]] clusters for a connected component graph."""
    if len(ccg) <= CC_MAX_SIZE:
        return [list(ccg.nodes())]
    # Lower resolution => fewer/bigger communities. You can expose as a flag if desired.
    # Start with 1.0; if you want *fewer* initial clusters globally, try 0.5 or 0.2.
    resolution = 1.0 if len(ccg) <= 10_000 else 0.6
    cs = nx.community.louvain_communities(ccg, seed=seed, resolution=resolution)
    return [list(c) for c in cs]

def build_initial_clusters(vocab_tokens, edges, seed=0):
    """Build global graph, split into CCs, run louvain (for big CCs), return clusters (list of lists)."""
    # Add self-loops for isolated vocab tokens
    edges_plus = edges + [(t, t) for t in vocab_tokens]

    graph = nx.Graph()
    graph.add_edges_from(edges_plus)

    # Make sure *every* vocab token is a node (even if not in any edge)
    graph.add_nodes_from(vocab_tokens)

    connected_components_node = list(nx.connected_components(graph))
    connected_components_node.sort(key=len, reverse=True)
    connected_components_graph = [graph.subgraph(ccn).copy() for ccn in connected_components_node]

    clusters = []
    for ccg in connected_components_graph:
        cs = louvain_or_trivial(ccg, seed=seed)
        clusters.extend(cs)

    # Filter to keep only tokens in vocab (defensive)
    vocab_set = set(vocab_tokens)
    valid_clusters = []
    for c in clusters:
        vc = [t for t in c if t in vocab_set]
        if vc:
            valid_clusters.append(vc)

    # Sanity check: each token must appear exactly once
    flat = [t for c in valid_clusters for t in c]
    assert len(flat) == len(vocab_tokens), f"Coverage mismatch: {len(flat)} vs {len(vocab_tokens)}"
    assert len(set(flat)) == len(vocab_tokens), "Duplicate token assignment found"
    return valid_clusters

def coarsen_clusters(clusters, target_k):
    """
    Greedy coarsening: merge smallest clusters until we have exactly target_k clusters.
    Keeps all tokens; just unions cluster lists. Deterministic given initial order.
    """
    if target_k <= 0:
        raise ValueError("target_k must be > 0")
    if target_k >= len(clusters):
        # Nothing to coarsen
        return [list(c) for c in clusters]

    # Min-heap by cluster size; store (size, idx, cluster_list)
    # idx breaks ties deterministically
    heap = []
    for idx, c in enumerate(clusters):
        heap.append((len(c), idx, c))
    heapq.heapify(heap)

    next_idx = len(clusters)
    while len(heap) > target_k:
        # Pop two smallest and merge
        s1, i1, c1 = heapq.heappop(heap)
        s2, i2, c2 = heapq.heappop(heap)
        merged = c1 + c2
        heapq.heappush(heap, (len(merged), next_idx, merged))
        next_idx += 1

    # Extract final clusters
    final = [c for (_, _, c) in heap]
    # Optional: sort by size desc for nicer stats/files
    final.sort(key=len, reverse=True)
    # Safety: ensure no token lost or duplicated relative to input clusters
    return final

def write_outputs(iter_idx, clusters, tokenizer, vocab_tokens, out_path_base, cluster_id_dim=300, rng=None):
    """
    Save token->clusterID mapping and clusters list into an iteration-indexed folder:
      <base_dir>/<iter_idx>/<basename>.json
      <base_dir>/<iter_idx>/<basename>_clusters.json

    `out_path_base` is your --output_file, e.g. $DATA_DIR/mapping/xsir/300_mapping_MODEL.json
    """
    if rng is None:
        rng = random.Random(0)

    # Map: token_id -> cluster_id (random label per cluster)
    mapping = [None] * len(vocab_tokens)
    cluster_ids = [rng.randint(0, cluster_id_dim - 1) for _ in range(len(clusters))]

    # Token -> id lookup once (vocab_tokens is list of tokens in tokenizer vocab order)
    tok2id = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in vocab_tokens}

    for ci, cluster in enumerate(clusters):
        cid = cluster_ids[ci]
        for tok in cluster:
            tid = tok2id[tok]
            # Some tokenizers can return None for weird special tokens; guard if needed
            assert tid is not None and 0 <= tid < len(mapping), f"Bad token id for {tok}: {tid}"
            assert mapping[tid] is None, f"Duplicate assignment for id {tid}"
            mapping[tid] = cid

    assert all(x is not None for x in mapping), "Incomplete token-to-cluster mapping"

    # Build iteration directory and filenames
    base_dir = os.path.dirname(out_path_base)          # e.g. $DATA_DIR/mapping/xsir
    base_name = os.path.basename(out_path_base)        # e.g. 300_mapping_MODEL.json
    iter_dir = os.path.join(base_dir, str(iter_idx))   # e.g. .../xsir/0
    os.makedirs(iter_dir, exist_ok=True)

    mapping_file = os.path.join(iter_dir, base_name)
    clusters_file = mapping_file.replace(".json", "_clusters.json")

    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)
    with open(clusters_file, "w") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    # Stats
    vocab_size = len(vocab_tokens)
    num_clusters = len(clusters)
    ge2 = sum(len(c) >= 2 for c in clusters)
    covered_by_ge2 = sum(len(c) for c in clusters if len(c) >= 2)
    top5 = [len(c) for c in clusters[:5]]
    print(f"[iter {iter_idx}] Vocabulary size: {vocab_size}")
    print(f"[iter {iter_idx}] Number of clusters: {num_clusters}")
    print(f"[iter {iter_idx}] Clusters with ≥2 tokens: {ge2}")
    print(f"[iter {iter_idx}] Vocab coverage by clusters ≥2 (%): {covered_by_ge2 / vocab_size * 100:.2f}")
    print(f"[iter {iter_idx}] Top 5 largest clusters: {top5}")
    print(f"[iter {iter_idx}] Saved: {mapping_file} and {clusters_file}")

def parse_targets(args, initial_k):
    """
    Determine the list of target cluster counts for iterations.
    Priority:
    1) --targets "60000,40000,20000"
    2) --fraction-steps "0.75,0.5,0.25" (apply to initial_k, floor to int and dedup/monotonic)
    3) Fallback: halve repeatedly until < 10k
    """
    if args.targets:
        targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    elif args.fraction_steps:
        fracs = [float(x.strip()) for x in args.fraction_steps.split(",") if x.strip()]
        # convert to counts
        counts = sorted({max(1, int(initial_k * f)) for f in fracs}, reverse=True)
        targets = counts
    else:
        # sensible default schedule
        t = []
        k = initial_k
        while k > 10_000:
            k = max(1, k // 2)
            t.append(k)
        targets = t

    # Ensure strictly decreasing and positive
    cleaned = []
    last = initial_k + 1
    for k in sorted(targets, reverse=True):
        if 0 < k < last:
            cleaned.append(k)
            last = k
    return cleaned

def main():
    parser = argparse.ArgumentParser(description="Generate token-to-cluster mappings with progressive coarsening.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dictionary", type=str, required=True, help="Dictionary file (one edge per line: src tgt)")
    parser.add_argument("--output_file", type=str, required=True, help="Base path for token->cluster mapping JSON (e.g., $DATA_DIR/mapping/xsir/300_mapping_MODEL.json)")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated target cluster counts, e.g., '60000,40000,20000'")
    parser.add_argument("--fraction_steps", type=str, default="", help="Comma-separated fractions of initial K, e.g., '0.75,0.5,0.25'")
    parser.add_argument("--cluster-id-dim", type=int, default=300, help="Upper bound for random cluster IDs (0..dim-1)")
    args = parser.parse_args()

    random.seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    # tokenizer.get_vocab() returns dict token->id, normalize to list indexed by id:
    vocab_by_id = [None] * len(vocab)
    for tok, tid in vocab.items():
        if 0 <= tid < len(vocab_by_id):
            vocab_by_id[tid] = tok
    # In case some gaps exist, fill with special placeholders to keep index order stable
    for i in range(len(vocab_by_id)):
        if vocab_by_id[i] is None:
            # Try to recover via tokenizer.convert_ids_to_tokens
            vocab_by_id[i] = tokenizer.convert_ids_to_tokens(i)

    # Load edges
    edges = []
    with open(args.dictionary, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            src_token, tgt_token = parts
            edges.append((src_token, tgt_token))

    # 1) Initial clustering
    base_clusters = build_initial_clusters(vocab_by_id, edges, seed=0)
    initial_k = len(base_clusters)
    print(f"[base] Initial number of clusters: {initial_k}")

    # Save the base (no coarsening) as iteration 0
    write_outputs(0, base_clusters, tokenizer, vocab_by_id, args.output_file, args.cluster_id_dim)

    # 2) Plan target steps
    targets = parse_targets(args, initial_k)
    if not targets:
        print("No coarsening targets specified or derived; only base files written.")
        return

    # 3) Coarsen iteratively and write files at each target (iterations 1..N)
    current = base_clusters
    for idx, k in enumerate(targets, start=1):
        current = coarsen_clusters(current, k)
        write_outputs(idx, current, tokenizer, vocab_by_id, args.output_file, args.cluster_id_dim)

if __name__ == "__main__":
    main()
