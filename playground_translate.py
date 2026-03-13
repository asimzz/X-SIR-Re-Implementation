#!/usr/bin/env python3
# ------------------------------------------------------------------
# translation_attack_grid.py
# ------------------------------------------------------------------
import os
import json
import math
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc

# ────────────────────────────────────────────────────────────────
# global style  (Times / Computer-Modern look)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.1,
    "grid.color": "grey",
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

# colours for the three seeds
SEED_COL = {"0": "C0", "42": "C1", "123": "C2"}

# ────────────────────────────────────────────────────────────────
def load_zscores(path):
    """Return a list of floats (z-scores) from a jsonl file."""
    with open(path) as f:
        return [
            float(json.loads(line)["z_score"]) if "z_score" in line else float(line)
            for line in f
        ]

def tpr_at_fpr(fpr, tpr, target=0.1):
    return float(interp1d(fpr, tpr, kind="linear", fill_value="extrapolate")(target))

# ────────────────────────────────────────────────────────────────
def plot_all_langs_grid(model_abbr, seeds, base_dir, out_png, tgt_langs):
    # grid: 5 columns, rows auto
    n_col, n_lang = 5, len(tgt_langs)
    n_row = math.ceil(n_lang / n_col)
    fig, axs = plt.subplots(
        n_row,
        n_col,
        figsize=(3.2 * n_col, 3.2 * n_row),
        sharex=True,
        sharey=True,
    )
    axs = axs.flatten()

    for idx, lang in enumerate(tgt_langs):
        ax = axs[idx]

        auc_rows, tpr_rows = [], []

        # ── draw ROC curves for each seed ──────────────────────
        for s in seeds:
            subdir = os.path.join(base_dir, model_abbr, f"xsir_seed{s}")
            hum = os.path.join(subdir, "mc4.en.hum.z_score.jsonl")
            atk = os.path.join(subdir, f"mc4.en-{lang}.mod.z_score.jsonl")
            if not (os.path.exists(hum) and os.path.exists(atk)):
                continue

            h = load_zscores(hum)
            a = load_zscores(atk)
            y_true = [0] * len(h) + [1] * len(a)
            y_score = h + a

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = auc(fpr, tpr)
            tpr_val = tpr_at_fpr(fpr, tpr)

            col = SEED_COL[s]
            ax.plot(fpr, tpr, lw=2.2, color=col)
            ax.plot(0.1, tpr_val, marker="o", ms=6, mfc="white", mew=2, color=col)

            auc_rows.append((s, auc_val))
            tpr_rows.append((s, tpr_val))

        # ── axis cosmetics ────────────────────────────────────
        ax.axvline(0.1, lw=1, color="grey", alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(lang)
        if idx // n_col == n_row - 1:
            ax.set_xlabel("FPR")
        if idx % n_col == 0:
            ax.set_ylabel("TPR")

        # ── inset white box  ----------------------------------
        ix, iy, iw, ih = 0.62, 0.04, 0.33, 0.42  # position & size
        ax.add_patch(
            FancyBboxPatch(
                (ix, iy),
                iw,
                ih,
                boxstyle="round,pad=0.25",
                fc="white",
                ec="black",
                lw=0.8,
                transform=ax.transAxes,
                zorder=2,
            )
        )

        # monospace font so decimals align
        def add_text(x, y, txt, **kw):
            ax.text(x, y, txt, transform=ax.transAxes, fontfamily="monospace", **kw)

        # headers
        add_text(ix + 0.04, iy + ih - 0.08, "AUC", fontsize=10, fontweight="bold", ha="left", va="top")
        add_text(ix + 0.04, iy + ih - 0.26, "TPR", fontsize=10, fontweight="bold", ha="left", va="top")

        # draw three rows under AUC
        row_y = iy + ih - 0.14
        for s, v in sorted(auc_rows, key=lambda x: int(x[0])):
            col = SEED_COL[s]
            # tiny dash
            ax.plot(
                [ix + 0.02, ix + 0.035],
                [row_y, row_y],
                transform=ax.transAxes,
                color=col,
                lw=2,
                solid_capstyle="butt",
            )
            add_text(ix + 0.04, row_y, f"{v:0.3f}", color=col, ha="left", va="center", fontsize=9)
            row_y -= 0.07

        # draw three rows under TPR
        row_y = iy + ih - 0.34
        for s, v in sorted(tpr_rows, key=lambda x: int(x[0])):
            col = SEED_COL[s]
            ax.plot(
                [ix + 0.02, ix + 0.035],
                [row_y, row_y],
                transform=ax.transAxes,
                color=col,
                lw=2,
                solid_capstyle="butt",
            )
            add_text(ix + 0.04, row_y, f"{v:0.3f}", color=col, ha="left", va="center", fontsize=9)
            row_y -= 0.07

    # hide unused axes
    for j in range(idx + 1, len(axs)):
        axs[j].set_visible(False)

    # global legend strip
    handles = [
        Line2D(
            [0], [0],
            color=SEED_COL[s],
            lw=2.2,
            marker="o",
            ms=6,
            mfc="white",
            mew=2,
        )
        for s in seeds
    ]
    labels = [f"seed {s}" for s in seeds]
    fig.legend(handles, labels, loc="upper center", ncol=len(seeds),
               frameon=True, fontsize=13, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_abbr", required=True)
    parser.add_argument("--base_dir",   required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    SEEDS = ["0", "42", "123"]
    LANGS = [
        "fr","de","it","es","pt",
        "pl","nl","ru","hi","ko","ja",
        "bn","fa","vi","iw","uk","ta"
    ]

    plot_all_langs_grid(
        args.model_abbr,
        SEEDS,
        args.base_dir,
        args.output,
        LANGS,
    )
