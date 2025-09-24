import argparse
import matplotlib.pyplot as plt

from utils import read_jsonl
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import json


ORG_LANGS = [
        "en", # English
        "it", # Italian
        "es", # Spanish
        "pt", # Portuguese
        "pl", # Polish
        "nl", # Dutch
        "hr", # Croatian
        "cs", # Czech
        "da", # Danish
        "ko", # Korean
        "ar"  # Arabic
    ]

def tpr_at_fpr(fpr, tpr, fpr_target):
    fpr_tpr_interpolation = interpolate.interp1d(fpr, tpr, kind="linear")
    return fpr_tpr_interpolation(fpr_target)


def f1_at_fpr(y_true, y_scores, fpr_target):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Finding the threshold for our target FPR
    threshold = thresholds[next(i for i in range(len(fpr)) if fpr[i] > fpr_target) - 1]
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)

    # Interpolating to find precision and recall at the target threshold
    precision_interp = interpolate.interp1d(
        thresholds_pr, precision[:-1], fill_value="extrapolate"
    )
    recall_interp = interpolate.interp1d(
        thresholds_pr, recall[:-1], fill_value="extrapolate"
    )
    precision_at_threshold = precision_interp(threshold)
    recall_at_threshold = recall_interp(threshold)

    # Calculate F1 score
    f1 = (
        2
        * (precision_at_threshold * recall_at_threshold)
        / (precision_at_threshold + recall_at_threshold)
    )

    return f1


def main(args):
    num_samples = 500
    true_lang = "en"
    
    
    wm_zscore = []
    candidate_hum_zscore = {}
    candidate_wm_zscore = {}
    tgt_lang = args.tgt_lang
    for lang in ORG_LANGS:
        if lang == tgt_lang:
            continue
        hum_zscore_file = args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.hum.z_score.jsonl"
        wm_zscore_file = args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.mod.z_score.jsonl"
        hum_list = read_jsonl(hum_zscore_file)
        wm_list = read_jsonl(wm_zscore_file)
        if len(hum_list) != len(wm_list):
            print(
                "The number of zscores in the human and watermark files are not the same."
            )
            return
        hum_zscore = [x["z_score"] if x["z_score"] is not None else 0 for x in hum_list]
        wm_zscore = [x["z_score"] if x["z_score"] is not None else 0 for x in wm_list]
        candidate_hum_zscore[lang] = hum_zscore
        candidate_wm_zscore[lang] = wm_zscore


    maximum_hum_zscore = []
    maximum_wm_zscore = []
    correct_hum_lang = 0
    correct_wm_lang = 0

    for i in range(num_samples):
        max_hum_score = float('-inf') 
        max_wm_score = float('-inf')
        best_hum_lang = None
        best_wm_lang = None
        for lang in ORG_LANGS:
            if lang == tgt_lang:
                continue
            hum_score = candidate_hum_zscore[lang][i]
            wm_score = candidate_wm_zscore[lang][i]
            if hum_score > max_hum_score:
                max_hum_score = hum_score
                best_hum_lang = lang
            if wm_score > max_wm_score:
                max_wm_score = wm_score
                best_wm_lang = lang
        maximum_hum_zscore.append(max_hum_score)
        maximum_wm_zscore.append(max_wm_score)
        if best_hum_lang == true_lang:
            correct_hum_lang += 1
        if best_wm_lang == true_lang:
            correct_wm_lang += 1

    print(f"Correct watermark language detection count: {correct_wm_lang}/{num_samples}")
    accuracy = (correct_wm_lang / num_samples) * 100
    print(f"Accuracy: {accuracy:.1f}")
    
    
    hm_true = [0 for _ in hum_list]
    wm_true = [1 for _ in wm_list]

    y_true = hm_true + wm_true
    y_scores = maximum_hum_zscore + maximum_wm_zscore

    auc = roc_auc_score(y_true, y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    print(
        f"""AUC: {auc:.3f}

TPR@FPR=0.1: {tpr_at_fpr(fpr, tpr, 0.1):.3f}
TPR@FPR=0.01: {tpr_at_fpr(fpr, tpr, 0.01):.3f}

F1@FPR=0.1: {f1_at_fpr(y_true, y_scores, 0.1):.3f}
F1@FPR=0.01: {f1_at_fpr(y_true, y_scores, 0.01):.3f}
"""
    )

    if args.roc_curve:
        with open(args.roc_curve, "w") as f:
            f.write(f"FPR\tTPR\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]:.3f}\t{tpr[i]:.3f}\n")

        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='red', linewidth=2)
        tpr_0_1 = tpr_at_fpr(fpr, tpr, 0.1)
        tpr_0_01 = tpr_at_fpr(fpr, tpr, 0.01)
        plt.scatter(0.1, tpr_0_1, color='red', s=40, edgecolors='black', label=f'TPR: {tpr_0_1:.3f}')
        plt.scatter(0.01, tpr_0_01, color='blue', s=40, edgecolors='black', label=f'TPR: {tpr_0_01:.3f}')
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc='lower right', fontsize=8, frameon=True)
        plt.grid()
        roc_curve_file = args.roc_curve.split(".txt")[0]
        plt.savefig(f"{roc_curve_file}.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate with watermarking")
    parser.add_argument(
        "--base_wm_dir", type=str, required=True, help="Base directory for watermark files"
    )
    parser.add_argument("--roc_curve", type=str, default=None, help="ROC curve file")
    parser.add_argument("--tgt_lang", type=str, default=None, help="Target language")

    args = parser.parse_args()
    main(args)
