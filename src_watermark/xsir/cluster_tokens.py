import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def count_tokens_in_valid_clusters(tokens, clusters_file):
    clusters = read_json(clusters_file)

    # Filter clusters with more than 2 tokens
    clusters = [cluster for cluster in clusters if len(cluster) >= 2]
    matching_tokens = []

    # Check each token against all clusters
    for token in tokens:
        for cluster in clusters:
            if token in cluster:
                matching_tokens.append(token)
                break  # No need to check other clusters once found
    token_count = len(matching_tokens)
    return token_count, matching_tokens


def main(args):
    # Load data
    input_data = read_jsonl(args.input_file)

    response_list = [d["response"] for d in input_data]
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    responses_tokens = [tokenizer.tokenize(text) for text in response_list]

    assert len(responses_tokens) == len(response_list), "Tokenization mismatch"
    nb_total_tokens = 0
    nb_matched_tokens = 0
    for tokens in responses_tokens:
        token_count, _ = count_tokens_in_valid_clusters(tokens, args.clusters_file)
        nb_total_tokens += len(tokens)
        nb_matched_tokens += token_count
    print(f"Total tokens: {nb_total_tokens}")
    print(f"Matched tokens: {nb_matched_tokens}")
    print(f"Percentage of matched tokens: {nb_matched_tokens / nb_total_tokens:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in valid clusters")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--clusters_file", type=str, required=True)

    args = parser.parse_args()
    main(args)
