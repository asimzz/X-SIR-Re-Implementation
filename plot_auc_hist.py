#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ─── Helpers ────────────────────────────────────────────────────────────────
def load_zscores(path):
    with open(path) as f:
        return [float(json.loads(line)["z_score"]) for line in f]

def compute_auc(hum_path, atk_path):
    h = load_zscores(hum_path)
    a = load_zscores(atk_path)
    y_true = [0]*len(h) + [1]*len(a)
    y_score = h + a
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot AUC per Seed bar-chart")
    parser.add_argument("--model_abbr", required=True)
    parser.add_argument("--base_dir",   required=True)
    parser.add_argument("--seeds",      nargs="+", required=True)
    parser.add_argument("--langs",      nargs="+", required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    # ─── Style ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size":        14,
        "axes.titlesize":   16,
        "axes.labelsize":   14,
        "xtick.labelsize":  10,
        "ytick.labelsize":  12,
        "axes.linewidth":   1.0,
    })

    seeds = sorted(int(s) for s in args.seeds)

    for lang in args.langs:
        # Gather (seed,auc) pairs
        data = []
        for seed in seeds:
            subdir   = os.path.join(args.base_dir, "xsir", f"seed_{seed}")
            hum_path = os.path.join(subdir, "mc4.en.hum.z_score.jsonl")
            atk_path = os.path.join(subdir, f"mc4.en-{lang}.mod.z_score.jsonl")
            if os.path.exists(hum_path) and os.path.exists(atk_path):
                try:
                    auc_val = compute_auc(hum_path, atk_path)
                    data.append((seed, auc_val))
                except Exception as e:
                    print(f"[Error] seed={seed}, lang={lang}: {e}")
            else:
                print(f"[Missing] seed={seed}, lang={lang}")

        if not data:
            print(f"No data for {lang}, skipping.")
            continue

        x, y = zip(*data)

        # ─── Plot ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        # Bars with no gap
        ax.bar(x, y,
               width=1.0,
               align="edge",
               color="royalblue",
               edgecolor="none")

        # ─── Labels / Cosmetics ─────────────────────────────────────────
        ax.set_title(f"AUC per Seed for “{lang}” ({args.model_abbr})", pad=14, weight="bold")
        ax.set_xlabel("Seed")
        ax.set_ylabel("AUC")
        ax.set_ylim(0, 1)
        ax.set_xlim(min(x), max(x)+1)     # ensure last bar fully visible
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90)
        ax.grid(axis="both", linestyle="--", alpha=0.4)

        plt.tight_layout()
        out_path = args.output.replace(".png", f"_{lang}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"→ saved {out_path}")

if __name__ == "__main__":
    main()
