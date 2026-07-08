#!/usr/bin/env python3
import json
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy import interpolate
import matplotlib.pyplot as plt

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate STEAM per-text results")
    parser.add_argument('--watermark_results', required=True, help='STEAM results for watermarked texts')
    parser.add_argument('--human_results', required=True, help='STEAM results for human texts')
    parser.add_argument('--roc_curve', help='Output ROC curve file')

    args = parser.parse_args()

    # Load results
    with open(args.watermark_results, 'r') as f:
        wm_results = json.load(f)

    with open(args.human_results, 'r') as f:
        hm_results = json.load(f)

    print(f"Loaded {len(wm_results)} watermarked and {len(hm_results)} human results")

    # Extract best z-scores for each text
    wm_zscores = [r.get('best_zscore', 0.0) for r in wm_results if r.get('best_lang')]
    hm_zscores = [r.get('best_zscore', 0.0) for r in hm_results if r.get('best_lang')]

    print(f"Successfully extracted {len(wm_zscores)} watermarked and {len(hm_zscores)} human z-scores")

    if len(wm_zscores) == 0 or len(hm_zscores) == 0:
        print("Error: No valid results found")
        return

    # Create labels and scores for ROC analysis
    y_true = [0] * len(hm_zscores) + [1] * len(wm_zscores)  # 0 = human, 1 = watermarked
    y_scores = hm_zscores + wm_zscores

    # Calculate AUC
    auc = roc_auc_score(y_true, y_scores)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate metrics
    tpr_01 = tpr_at_fpr(fpr, tpr, 0.1)
    tpr_001 = tpr_at_fpr(fpr, tpr, 0.01)
    f1_01 = f1_at_fpr(y_true, y_scores, 0.1)
    f1_001 = f1_at_fpr(y_true, y_scores, 0.01)

    print(f"""STEAM Per-Text Results:

AUC: {auc:.3f}

TPR@FPR=0.1: {tpr_01:.3f}
TPR@FPR=0.01: {tpr_001:.3f}

F1@FPR=0.1: {f1_01:.3f}
F1@FPR=0.01: {f1_001:.3f}

Average human z-score: {np.mean(hm_zscores):.3f}
Average watermark z-score: {np.mean(wm_zscores):.3f}
""")

    # Save ROC curve if requested
    if args.roc_curve:
        with open(args.roc_curve, "w") as f:
            f.write(f"FPR\tTPR\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]:.3f}\t{tpr[i]:.3f}\n")

        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='red', linewidth=2)
        plt.scatter(0.1, tpr_01, color='red', s=40, edgecolors='black', label=f'TPR@0.1: {tpr_01:.3f}')
        plt.scatter(0.01, tpr_001, color='blue', s=40, edgecolors='black', label=f'TPR@0.01: {tpr_001:.3f}')
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc='lower right', fontsize=8, frameon=True)
        plt.grid()
        roc_curve_file = args.roc_curve.split(".txt")[0]
        plt.savefig(f"{roc_curve_file}.png", dpi=300, bbox_inches='tight')

        print(f"ROC curve saved to {args.roc_curve} and {roc_curve_file}.png")

    # Print best languages distribution
    wm_best_langs = [r.get('best_lang') for r in wm_results if r.get('best_lang')]
    hm_best_langs = [r.get('best_lang') for r in hm_results if r.get('best_lang')]

    from collections import Counter
    wm_lang_counts = Counter(wm_best_langs)
    hm_lang_counts = Counter(hm_best_langs)

    print("\nTop 5 best languages for watermarked texts:")
    for lang, count in wm_lang_counts.most_common(5):
        print(f"  {lang}: {count} texts ({count/len(wm_best_langs)*100:.1f}%)")

    print("\nTop 5 best languages for human texts:")
    for lang, count in hm_lang_counts.most_common(5):
        print(f"  {lang}: {count} texts ({count/len(hm_best_langs)*100:.1f}%)")

if __name__ == "__main__":
    main()