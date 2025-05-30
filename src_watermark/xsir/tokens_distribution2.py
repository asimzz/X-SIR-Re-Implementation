import argparse
from transformers import AutoTokenizer
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def plot_token_distribution(tokens_list, base_model, tgt_lang, seed,top_percent=80,top_k=10):
    # Count token frequency
    total_tokens = len(tokens_list)
    token_counts = Counter(tokens_list)

    # Convert to percentage
    token_percentages = {token: (count / total_tokens) * 100 for token, count in token_counts.items()}
    sorted_tokens = sorted(token_percentages.items(), key=lambda x: x[1], reverse=True)

    # Compute number of tokens to cover top_percent%
    cumulative = 0.0
    selected_tokens = []
    for token, pct in sorted_tokens:
        if cumulative >= top_percent:
            break
        selected_tokens.append((token, pct))
        cumulative += pct

    tokens, percentages = zip(*selected_tokens)
    print(f"Number of tokens covering top {top_percent}%: {len(tokens)}")
    print(f"Summation of selected token percentages: {sum(percentages):.2f}% of total tokens")
    print(f"Total tokens: {total_tokens}")
    # sns.barplot(y=list(tokens[:top_k]), x=list(percentages[:top_k]), hue=list(tokens[:top_k]), palette="viridis", legend=False)
    # plt.xlabel("Percentage (%)")
    # plt.ylabel("Token")
    # plt.title("Token Distribution")
    # plt.title(f"Top {top_k} Token Distribution for {base_model} in {tgt_lang} with seed {seed}")
    # plt.show()
    # return


def main(args):
    t_data = read_jsonl(args.translation_file)
    tokens_biases_list = [d["biases"] for d in t_data]
    
    tokens_list = []
    for tokens_biases in tokens_biases_list:
        tokens = [bias[0] for bias in tokens_biases]
        tokens_list.extend(tokens)

    plot_token_distribution(tokens_list, args.base_model, args.tgt_lang, args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Distribution Plotter")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--translation_file", type=str, required=True)

    args = parser.parse_args()
    main(args)
