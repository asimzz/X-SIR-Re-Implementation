import argparse
from transformers import AutoTokenizer
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def plot_token_distribution(tokens_list, base_model, top_k=10):
    # Count token frequency
    total_tokens = len(tokens_list)
    token_counts = Counter(tokens_list)

    # Convert to percentage
    token_percentages = {token: (count / total_tokens) * 100 for token, count in token_counts.items()}
    most_common = sorted(token_percentages.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Prepare data for plotting
    tokens, percentages = zip(*most_common)
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(14, 7))
    sns.barplot(x=list(tokens), y=list(percentages), palette="viridis")

    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Token", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title(f"Top {top_k} Token Percentage Distribution for {base_model}", fontsize=14)
    plt.tight_layout()
    plt.show()


def main(args):
    t_data = read_jsonl(args.translation_file)
    tokens_biases_list = [d["biases"] for d in t_data]
    
    tokens_list = []
    for tokens_biases in tokens_biases_list:
        tokens = [bias[0] for bias in tokens_biases]
        tokens_list.extend(tokens)

    plot_token_distribution(tokens_list, args.base_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Distribution Plotter")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--translation_file", type=str, required=True)

    args = parser.parse_args()
    main(args)
