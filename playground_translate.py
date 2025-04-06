import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_jsonl
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve

# Use a modern seaborn style
sns.set_style("whitegrid")
sns.set_palette("colorblind")
sns.set_context("talk")


def tpr_at_fpr(fpr, tpr, fpr_target):
    fpr_tpr_interpolation = interpolate.interp1d(
        fpr, tpr, kind="linear", fill_value="extrapolate"
    )
    return fpr_tpr_interpolation(fpr_target)


def process_file(file_path):
    wm_list = read_jsonl(file_path)
    wm_zscore = [x["z_score"] if x["z_score"] is not None else 0 for x in wm_list]
    wm_true = [1 for x in wm_list]
    return wm_zscore, wm_true


def main(args):
    hm_list = read_jsonl(args.hm_zscore)[:100]
    hm_zscore = [x["z_score"] if x["z_score"] is not None else 0 for x in hm_list]
    hm_true = [0 for x in hm_list]

    attack_types = [
        (args.wm_no_translation_zscore, "No Translation Attack"),
        (args.wm_translation_attack_zscore, "Translation Attack"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    for attack_file, label in attack_types:
        wm_zscore, wm_true = process_file(attack_file)
        y_true = hm_true + wm_true
        y_scores = hm_zscore + wm_zscore

        auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        (line,) = ax.plot(
            fpr, tpr, linewidth=2.5, label=f"AUC = {auc:.3f}", alpha=0.9
        )

        # TPR at FPR = 0.1 marker
        ax.scatter(
            0.1,
            tpr_at_fpr(fpr, tpr, 0.1),
            s=100,
            color="white",
            edgecolors=line.get_color(),
            linewidth=2.5,
            label=f"TPR@FPR=0.1"
        )

    model_name = args.model_abbr
    tgt_lang = args.tgt_lang
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=14)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=14)
    ax.set_title(f"ROC Curve - {model_name} - {tgt_lang}", fontsize=16, fontweight="bold")
    ax.legend(
        loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True
    )
    ax.grid(True, linestyle="--", alpha=0.4)

    if args.roc_curve:
        plt.savefig(args.roc_curve, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ROC curves for multiple models and attacks"
    )
    parser.add_argument(
        "--hm_zscore", type=str, required=True, help="Path to human z-score file"
    )
    parser.add_argument(
        "--wm_no_translation_zscore",
        type=str,
        required=True,
        help="Path to No Translation Attack z-score file",
    )
    parser.add_argument(
        "--wm_translation_attack_zscore",
        type=str,
        required=True,
        help="Path to Translation Attack z-score file",
    )
    parser.add_argument(
        "--translation_attack_file",
        type=str,
        required=True,
        help="Path to CWRA attack file",
    )
    parser.add_argument(
        "--no_translation_attack_file",
        type=str,
        required=True,
        help="Path to No CWRA attack file",
    )
    
    parser.add_argument(
        "--model_abbr", type=str, required=True, help="Model abbreviation"
    )
    
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")

    parser.add_argument(
        "--roc_curve", type=str, default=None, help="Output ROC curve file"
    )
    parser.add_argument(
        "--figure_title", type=str, default=None, help="Title of the figure"
    )

    args = parser.parse_args()
    main(args)
