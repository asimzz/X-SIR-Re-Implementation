import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import interpolate
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import argparse

# Set global style
sns.set(style="whitegrid", context="notebook", palette="colorblind")
plt.rc('font', size=14)
plt.rc('axes', labelsize=13)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "font.size": 13,
})

def tpr_at_fpr(fpr, tpr, fpr_target):
    f = interpolate.interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")
    return float(f(fpr_target))

def load_zscores(file_path):
    scores = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            score = data.get("z_score", None)
            if score is not None:
                scores.append(float(score))
            else:
                try:
                    scores.append(float(data))
                except:
                    raise ValueError(f"Invalid score in file: {file_path}")
    return scores

def plot_model_roc_per_lang(model_abbr, seeds, base_dir, output_path, tgt_langs):
    seed_color_map = {
        "0": "C0",
        "42": "C1",
        "123": "C2"
    }

    for tgt_lang in tgt_langs:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')

        for seed in seeds:
            input_dir = os.path.join(base_dir, model_abbr, f"xsir_seed{seed}")
            human_file = os.path.join(input_dir, "mc4.en.hum.z_score.jsonl")
            attack_file = os.path.join(input_dir, f"mc4.en-{tgt_lang}.mod.z_score.jsonl")

            if not os.path.exists(attack_file) or not os.path.exists(human_file):
                continue

            human_scores = load_zscores(human_file)
            attack_scores = load_zscores(attack_file)
            y_true = [0]*len(human_scores) + [1]*len(attack_scores)
            y_score = human_scores + attack_scores
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = auc(fpr, tpr)
            tpr_marker = tpr_at_fpr(fpr, tpr, 0.1)

            color = seed_color_map[seed]
            ax.plot(fpr, tpr, label=f"Seed {seed} (AUC={auc_val:.3f})", color=color, linewidth=2.8)
            ax.plot(0.1, tpr_marker, 'o', markersize=7, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=2)

        ax.axvline(x=0.1, linestyle='-', color='gray', alpha=0.3, linewidth=1.2)
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title(f"Translation Attack ROC ({tgt_lang}) on {model_abbr}", fontsize=15)
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        ax.grid(True, axis='y', linestyle="--", alpha=0.4)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(output_path, f"roc_{model_abbr}_{tgt_lang}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_abbr", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    seeds = ["0", "42", "123"]
    tgt_langs = ["it", "es", "pt", "tr", "ar", "sw", "am"]
    os.makedirs(args.output, exist_ok=True)
    plot_model_roc_per_lang(args.model_abbr, seeds, args.base_dir, args.output, tgt_langs)
