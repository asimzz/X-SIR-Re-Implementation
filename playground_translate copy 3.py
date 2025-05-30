import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import interpolate
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import argparse

# ─── Global Style ─────────────────────────────────────────────────────────────
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

# ─── Helpers ──────────────────────────────────────────────────────────────────
def tpr_at_fpr(fpr, tpr, fpr_target=0.1):
    interp = interpolate.interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")
    return float(interp(fpr_target))

def load_zscores(path):
    scores = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            z = data.get("z_score")
            if z is not None:
                scores.append(float(z))
            else:
                # fallback if the file is just numbers
                scores.append(float(data))
    return scores

# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_all_langs_grid(model_abbr, seeds, base_dir, out_path, tgt_langs):
    # assign one distinct color per seed
    seed_colors = {"0":"C0","42":"C1","123":"C2"}

    # grid: 2 rows × 4 cols
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    for ax_idx, lang in enumerate(tgt_langs):
        ax = axs[ax_idx]
        for seed in seeds:
            inp = os.path.join(base_dir, model_abbr, f"xsir_seed{seed}")
            hum = os.path.join(inp, "mc4.en.hum.z_score.jsonl")
            atk = os.path.join(inp, f"mc4.en-{lang}.mod.z_score.jsonl")
            if not (os.path.exists(hum) and os.path.exists(atk)):
                continue

            h_scores = load_zscores(hum)
            a_scores = load_zscores(atk)
            y_true  = [0]*len(h_scores) + [1]*len(a_scores)
            y_score = h_scores + a_scores

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_v = auc(fpr, tpr)
            tpr_marker = tpr_at_fpr(fpr, tpr, 0.1)

            color = seed_colors[seed]

            # style each subplot
            header_style = dict(color='white', linewidth=0)

            inner_legend = [
                (Line2D([0], [-0.5], **header_style), "AUC"),
                (Line2D([0], [0], color=color, lw=2), f"{auc_v:.3f}"),
                (Line2D([0], [0], **header_style), "TPR"),
                (Line2D([0], [0], color=color, lw=0, marker='o', markerfacecolor='white',
                        markeredgecolor=color, markersize=6), f"{tpr_marker:.3f}"),
            ]

            handles, labels = zip(*inner_legend)
            ax.legend(handles, labels, loc="lower right", fontsize=10, frameon=True, handlelength=2)
            # Add subplot annotation like (b1), (b2), ...
            subplot_label = f"(b{ax_idx + 1})  Seed {seed}"
            ax.text(0.5, 1.1, subplot_label, transform=ax.transAxes,
                    ha='center', va='top', fontsize=12)
            # ax.set_title(f"Seed {seed}", fontsize=13)
            ax.set_xlabel("False Positive Rate (FPR)")
            if ax_idx == 0:
                ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(False)
        ax.grid(axis='y', linestyle="-", alpha=0.4)  # Keep horizontal grid lines
        ax.axvline(x=0.1, linestyle='-', color='gray', alpha=0.2, linewidth=1.2) 
        ax.axvline(0.1, color='gray', alpha=0.3, lw=1.2)
        ax.grid(axis='y', linestyle="--", alpha=0.4)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f"{lang}", fontsize=14)
        if ax_idx>=4:  ax.set_xlabel("FPR")
        if ax_idx%4==0: ax.set_ylabel("TPR")

    # hide the extra (8th) subplot
    axs[-1].axis('off')

    # shared legend at bottom
    handles = [Line2D([0],[0],color=seed_colors[s],lw=2.5) for s in seeds]
    labels  = [f"seed {s}" for s in seeds]
    fig.legend(handles, labels,
               loc='lower center', ncol=len(seeds),
               frameon=True, fontsize=12, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(f"Translation‐attack ROC curves on {model_abbr}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0,0.05,1,0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_abbr", type=str, required=True)
    p.add_argument("--base_dir",   type=str, required=True)
    p.add_argument("--output",     type=str, required=True)
    args = p.parse_args()

    SEEDS     = ["0","42","123"]
    LANGS     = ["it","es","pt","tr","ar","sw","am"]
    plot_all_langs_grid(
        args.model_abbr,
        SEEDS,
        args.base_dir,
        args.output,
        LANGS
    )
