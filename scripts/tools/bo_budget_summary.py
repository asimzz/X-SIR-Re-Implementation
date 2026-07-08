import argparse
import csv
import os
import sys

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from bo_budget_csv import BUDGETS, LANGUAGES, bo_paths, extract_zscores
from eval_detection import tpr_at_fpr
from utils import read_jsonl


def per_lang_roc(wm_path, hum_path):
    wm = read_jsonl(wm_path)
    hum = read_jsonl(hum_path)
    n = min(len(wm), len(hum))
    wm, hum = wm[:n], hum[:n]
    y_true = [0] * len(hum) + [1] * len(wm)
    y_scores = extract_zscores(hum) + extract_zscores(wm)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    tpr1 = float(tpr_at_fpr(fpr, tpr, 0.01))
    return fpr, tpr, auc, tpr1


def macro_average_roc(rocs, grid):
    interp_tprs = []
    for fpr, tpr in rocs:
        interp_tprs.append(np.interp(grid, fpr, tpr))
    return np.mean(interp_tprs, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Summary table (mean +/- std) and macro-averaged ROC overlay across BO budgets."
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.join(REPO_ROOT, "gen/aya-23-8B/kgw_seed0"),
    )
    parser.add_argument("--out-csv", default="bo_budget_summary.csv")
    parser.add_argument("--out-plot", default="bo_budget_roc.png")
    args = parser.parse_args()

    grid = np.linspace(0.0, 1.0, 1001)
    per_budget_aucs = {b: [] for b in BUDGETS}
    per_budget_tpr1 = {b: [] for b in BUDGETS}
    per_budget_rocs = {b: [] for b in BUDGETS}

    for budget in BUDGETS:
        for lang in LANGUAGES:
            wm_path, hum_path = bo_paths(args.base_dir, lang, budget)
            if not (os.path.exists(wm_path) and os.path.exists(hum_path)):
                print(
                    f"warning: missing files for {lang} budget={budget}",
                    file=sys.stderr,
                )
                continue
            fpr, tpr, auc, tpr1 = per_lang_roc(wm_path, hum_path)
            per_budget_aucs[budget].append(auc)
            per_budget_tpr1[budget].append(tpr1)
            per_budget_rocs[budget].append((fpr, tpr))

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["budget", "n_languages", "auc_mean", "auc_std", "tpr_at_1pct_mean", "tpr_at_1pct_std"]
        )
        for budget in BUDGETS:
            aucs = np.array(per_budget_aucs[budget])
            tprs = np.array(per_budget_tpr1[budget])
            writer.writerow(
                [
                    budget,
                    len(aucs),
                    f"{aucs.mean():.6f}",
                    f"{aucs.std(ddof=1):.6f}",
                    f"{tprs.mean():.6f}",
                    f"{tprs.std(ddof=1):.6f}",
                ]
            )
    print(f"wrote {args.out_csv}")

    cb_palette = ["#0072B2", "#DE8F05", "#029E73", "#D55E00"]
    colors = {b: c for b, c in zip(BUDGETS, cb_palette)}
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    curve_handles = []
    auc_handles = []
    tpr_handles = []
    for budget in BUDGETS:
        mean_tpr = macro_average_roc(per_budget_rocs[budget], grid)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        macro_auc = float(np.mean(per_budget_aucs[budget]))
        macro_tpr1 = float(np.interp(0.01, grid, mean_tpr))
        (line,) = ax.plot(
            grid,
            mean_tpr,
            color=colors[budget],
            linewidth=2.5,
            label=f"budget={budget}",
            zorder=10,
            clip_on=False,
        )
        ax.scatter(
            [0.01],
            [macro_tpr1],
            facecolors="white",
            edgecolors=colors[budget],
            s=40,
            linewidths=1.6,
            zorder=11,
            clip_on=False,
        )
        curve_handles.append(line)
        auc_handles.append(
            mlines.Line2D([], [], color=colors[budget], linewidth=2, label=f"{macro_auc:.3f}")
        )
        tpr_handles.append(
            mlines.Line2D(
                [],
                [],
                color=colors[budget],
                marker="o",
                markerfacecolor="white",
                markeredgecolor=colors[budget],
                markeredgewidth=1.6,
                markersize=6,
                linestyle="none",
                label=f"{macro_tpr1:.3f}",
            )
        )

    ax.axvline(0.01, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.margins(0)
    ax.grid(False)
    ax.yaxis.grid(True, color="lightgray", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    top_legend = fig.legend(
        handles=curve_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        frameon=True,
        fontsize=10,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.2,
        borderpad=0.4,
    )
    top_legend.get_frame().set_boxstyle("round,pad=0.3")
    top_legend.get_frame().set_edgecolor("#888888")

    auc_legend = ax.legend(
        handles=auc_handles,
        loc="lower right",
        bbox_to_anchor=(0.62, 0.02),
        title="AUC",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.3,
        labelspacing=0.25,
    )
    auc_legend.get_frame().set_boxstyle("round,pad=0.2")
    auc_legend.get_frame().set_facecolor("none")
    auc_legend.get_frame().set_edgecolor("#888888")

    tpr_legend = ax.legend(
        handles=tpr_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        title="TPR@1%",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.3,
        labelspacing=0.25,
    )
    tpr_legend.get_frame().set_boxstyle("round,pad=0.2")
    tpr_legend.get_frame().set_facecolor("none")
    tpr_legend.get_frame().set_edgecolor("#888888")
    ax.add_artist(auc_legend)

    plt.savefig(args.out_plot, dpi=300, bbox_inches="tight", bbox_extra_artists=[top_legend])
    print(f"wrote {args.out_plot}")
    pdf_path = os.path.splitext(args.out_plot)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight", bbox_extra_artists=[top_legend])
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
