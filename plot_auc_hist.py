#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import seaborn as sns
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
    parser = argparse.ArgumentParser(description="Plot AUC per Seed bar‐chart")
    parser.add_argument("--model_abbr", required=True)
    parser.add_argument("--base_dir",   required=True)
    parser.add_argument("--seeds",      nargs="+", required=True)
    parser.add_argument("--langs",      nargs="+", required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    # ─── LaTeX‐style rcParams ────────────────────────────────────────────────
    plt.rcParams.update({
        "text.usetex":     False,                      # no external TeX call
        "font.family":     "serif",
        "font.serif":      ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize":  12,
        "axes.labelsize":  10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth":  0.8,
        "axes.edgecolor":  "black",
        "grid.color":      "gray",
        "grid.linestyle":  "--",
        "grid.alpha":      0.4,
        "figure.dpi":      300,
    })
    sns.set_style("white")  # clean white background

    seeds = sorted(int(s) for s in args.seeds)

    for lang in args.langs:
        # collect (seed, auc)
        data = []
        for seed in seeds:
            subdir   = os.path.join(args.base_dir, "xsir", f"seed_{seed}")
            hum_path = os.path.join(subdir, "mc4.en.hum.z_score.jsonl")
            atk_path = os.path.join(subdir, f"mc4.en-{lang}.mod.z_score.jsonl")
            if os.path.exists(hum_path) and os.path.exists(atk_path):
                try:
                    data.append((seed, compute_auc(hum_path, atk_path)))
                except Exception as e:
                    print(f"[Error] seed={seed}, lang={lang}: {e}")
            else:
                print(f"[Missing] seed={seed}, lang={lang}")

        if not data:
            print(f"No data for {lang}, skipping.")
            continue

        x, y = zip(*data)

        # ─── Plot ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            y, 
            bins=10, 
            color="C0", 
            edgecolor="black", 
            alpha=0.85
        )

        # ─── Cosmetics ───────────────────────────────────────────────────
        ax.set_title(f"AUC Distribution for “{lang}” ({args.model_abbr})")
        ax.set_xlabel("AUC")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y")               # only horizontal grid
        # remove top & right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out_path = args.output.replace(".png", f"_{lang}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"→ saved {out_path}")

if __name__ == "__main__":
    main()
