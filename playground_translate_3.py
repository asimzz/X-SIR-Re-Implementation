import os
import math
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
plt.rc("font", size=14)
plt.rc("axes", labelsize=13)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "font.size": 14,
    }
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def tpr_at_fpr(fpr, tpr, fpr_target=0.1):
    interp = interpolate.interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")
    return float(interp(fpr_target))


def load_zscores(path):
    scores = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            z = data.get("z_score")
            # print(f"Loading z-score from {path}: {z}")
            if z is not None:
                scores.append(float(z))
            else:
                # fallback if the file is just numbers
                scores.append(float(0))
    return scores


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_all_langs_grid(model_abbr, seeds, base_dir, out_path, tgt_langs, org_lang="en"):
    # assign one distinct color per seed
    seed_colors = {"0": "C0", "42": "C1", "123": "C2"}

    n_langs = len(tgt_langs)
    n_cols = 4
    n_rows = math.ceil(n_langs / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.75, n_rows * 3.05), sharex=True, sharey=True)
    axs = axs.flatten()

    for ax_idx, lang in enumerate(tgt_langs):
        if org_lang == lang:
            continue
        ax = axs[ax_idx]
        inner_legend = []
        header_style = dict(color="white", linewidth=0)
        auc_legend = [(Line2D([0], [-0.5], **header_style), "AUC")]
        tpr_legend = [(Line2D([0], [0], **header_style), "TPR")]

        def plot_roc(ax, atk, hum, auc_legend, tpr_legend, inner_legend, color):
            if not (os.path.exists(hum) and os.path.exists(atk)):
                return

            h_scores = load_zscores(hum)
            a_scores = load_zscores(atk)
            y_true = [0] * len(h_scores) + [1] * len(a_scores)
            y_score = h_scores + a_scores

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_v = auc(fpr, tpr)
            tpr_marker = tpr_at_fpr(fpr, tpr, 0.1)

            ax.plot(fpr, tpr, lw=3, color=color)
            ax.plot(0.1, tpr_marker, 'o', markersize=9, markerfacecolor='white', lw=2, color=color,
                 markeredgecolor=color)

            auc_legend += [
                (Line2D([0], [0], color=color, lw=3), f"{auc_v:.3f}"),
            ]
            tpr_legend += [
                (
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        lw=3,
                        marker="o",
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markersize=6,
                    ),
                    f"{tpr_marker:.3f}",
                ),
            ]

            inner_legend = auc_legend + tpr_legend

            handles, labels = zip(*inner_legend)
            ax.legend(
                handles,
                labels,
                loc="lower right",
                fontsize=10,
                frameon=True,
                handlelength=2,
            )
            ax.set_xlabel("False Positive Rate (FPR)")
            ax.set_ylabel("True Positive Rate (TPR)")

            # style each subplot

        for i, seed in enumerate(seeds):
            inp = os.path.join(base_dir, model_abbr, f"kgw")
            hum = os.path.join(inp, f"mc4.en.hum.z_score.jsonl")
            atk = os.path.join(inp, f"mc4.en-{lang}.mod.z_score.jsonl")
            plot_roc(ax, atk, hum, auc_legend, tpr_legend, inner_legend, seed_colors[seed])

        # Google back-translation
        back_translated = os.path.join(inp, f"mc4.{lang}-{org_lang}-back.mod.z_score.jsonl")
        plot_roc(ax, back_translated, hum, auc_legend, tpr_legend, inner_legend, "black")   

        # style each subplot

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(False)
        ax.grid(axis="y", linestyle="-", alpha=0.4)  # Keep horizontal grid lines
        ax.axvline(x=0.1, linestyle="-", color="gray", alpha=0.2, linewidth=1.2)
        ax.axvline(0.1, color="gray", alpha=0.3, lw=1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{lang}", fontsize=14)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)

    # hide the extra (8th) subplot
    for ax in axs[n_langs:]:
        ax.axis("off")

    # shared legend at bottom
    handles = [Line2D([0], [0], color=seed_colors[s], lw=2.5) for s in seeds]
    handles = [Line2D([0], [0], color="black", lw=2.5)] + handles  # Add back-translation color
    handles = [Line2D([0], [0], color="red", lw=2.5)] + handles  # Add back-translation color
    labels = [f"seed {s}" for s in seeds]
    labels =  [ "Deepseek Back-translation", "Google Back-translation", "Without Back-translation"]  # Add back-translation label
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=True,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle(f"Transaltion-attack ROC curves on {model_abbr} (KGW)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_abbr", type=str, required=True)
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--org_lang", type=str, default="en",)
    args = p.parse_args()

    SEEDS = ["0"]
    LANGS = [ 
    # High-resource languages
    "fr",
    "de",
    "it",
    "es",
    "pt",
    # Medium-resource languages
    "pl",
    "nl",
    "ru",
    "hi",
    "ko",
    "ja",
    # # Low-resource languages
    "bn",
    "fa",
    "vi",
    "iw", # Hebrew
    "uk",
    "ta"
    ]
plot_all_langs_grid(args.model_abbr, SEEDS, args.base_dir, args.output, LANGS, org_lang=args.org_lang)
