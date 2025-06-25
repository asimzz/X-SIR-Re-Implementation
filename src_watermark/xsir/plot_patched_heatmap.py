#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_zscores(path):
    """Read a .jsonl where each line is either a float or {"z_score":…}."""
    with open(path) as f:
        out = []
        for line in f:
            j = json.loads(line)
            if isinstance(j, dict) and "z_score" in j:
                out.append(float(j["z_score"]))
            else:
                out.append(float(j))
        return out

def compute_auc(hum_path, atk_path):
    h = load_zscores(hum_path)
    a = load_zscores(atk_path)
    y_true = [0]*len(h) + [1]*len(a)
    y_score = h + a
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def main():
    p = argparse.ArgumentParser(
        description="Plot a heatmap of AUCs when patching top‐10 tokens from ref seeds"
    )
    p.add_argument("--model_abbr", required=True)
    p.add_argument("--base_dir",   required=True,
                   help="gen/<model_abbr>/xsir/seed_<ref_seed>/…")
    p.add_argument("--seeds",      nargs="+", required=True,
                   help="List of four seeds, e.g. 0 1 10 11")
    p.add_argument("--tgt_lang",   required=True,
                   help="Target language code, e.g. bn")
    p.add_argument("--output",     required=True,
                   help="Where to save the heatmap PNG")
    args = p.parse_args()

    # ensure integer seeds, sort
    seeds = sorted(int(s) for s in args.seeds)
    n = len(seeds)

    # allocate matrix
    mat = np.zeros((n, n), dtype=float)

    for i, ref in enumerate(seeds):
        for j, patch in enumerate(seeds):
            subdir = os.path.join(
                args.base_dir,
                args.model_abbr,
                "xsir",
                f"seed_{ref}"
            )
            hum_fn = "mc4.en.hum.z_score.jsonl"
            if ref == patch:
                atk_fn = f"mc4.en-{args.tgt_lang}.mod.z_score.jsonl"
            else:
                atk_fn = f"mc4.en-{args.tgt_lang}-seed-{patch}.mod.z_score.jsonl"

            hum_path = os.path.join(subdir, hum_fn)
            atk_path = os.path.join(subdir, atk_fn)

            if not (os.path.exists(hum_path) and os.path.exists(atk_path)):
                print(f"[WARN] missing files for ref={ref}, patch={patch}")
                mat[i,j] = np.nan
            else:
                try:
                    mat[i,j] = compute_auc(hum_path, atk_path)
                except Exception as e:
                    print(f"[ERROR] ref={ref}, patch={patch}: {e}")
                    mat[i,j] = np.nan

     # ─── LaTeX-style rcParams ───────────────────────────────────────
    plt.rcParams.update({
        "text.usetex":    False,                    # no external tex
        "font.family":    "serif",
        "font.serif":     ["Computer Modern Roman"],
        "mathtext.fontset":"cm",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "figure.dpi":     300,
    })
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6, 5))
    # no automatic annot
    im = sns.heatmap(
        mat,
        annot=False,
        cmap="crest",
        cbar_kws={"label": "AUC"},
        xticklabels=seeds,
        yticklabels=seeds,
        ax=ax,
        square=True
    )
    # annotate only diagonal
    for idx in range(n):
        val = mat[idx, idx]
        if not np.isnan(val):
            ax.text(
                idx + 0.5,           # x
                idx + 0.5,           # y
                f"{val:.3f}",        # text
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontfamily="monospace"
            )

    ax.set_xlabel("Patched Seed")
    ax.set_ylabel("Reference Seed")
    ax.set_title(
        f"AUC Heatmap (patch top-10 tokens from ref seed)\n"
        f"{args.model_abbr} – {args.tgt_lang}"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"Saved heatmap to {args.output}")

if __name__ == "__main__":
    main()
