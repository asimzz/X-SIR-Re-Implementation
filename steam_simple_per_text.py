#!/usr/bin/env python3
import argparse
import json
import numpy as np
from utils import read_jsonl
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

ORG_LANGS = [
    "en", # English
    # High-resource languages
    "fr", # French
    "de", # German
    "it", # Italian
    "es", # Spanish
    "pt", # Portuguese
    # Medium-resource languages
    "pl", # Polish
    "nl", # Dutch
    "ru", # Russian
    "hi", # Hindi
    "ko", # Korean
    "ja", # Japanese
    # Low-resource languages
    "bn", # Bengali
    "fa", # Persian
    "vi", # Vietnamese
    "iw", # Hebrew
    "uk", # Ukrainian
    "ta", # Tamil
]

def get_zscore_for_text(data_list, text_idx):
    """Extract z-score for specific text index."""
    if text_idx < len(data_list):
        item = data_list[text_idx]
        return item["z_score"] if item["z_score"] is not None else 0
    return 0

def get_avg_zscore(validation_file):
    """Get average z-score from validation file."""
    validation_list = read_jsonl(validation_file)
    zscores = [item["z_score"] if item["z_score"] is not None else 0 for item in validation_list]
    return sum(zscores) / len(zscores) if zscores else 0

def steam_per_text_simple(base_dir, target_lang, num_samples=500):
    """
    Simple per-text STEAM evaluation like the batch approach but applied per text.

    For each text:
    1. Get z-scores for all intermediate languages
    2. Normalize using validation
    3. Find best intermediate language for this specific text
    4. Return the maximum normalized z-score
    """

    results = []

    # Load suspect attack files (the main evaluation files)
    suspect_attack_wm_file = f"{base_dir}/mc4.en-{target_lang}.mod.z_score.jsonl"
    suspect_attack_hum_file = f"{base_dir}/mc4.en-{target_lang}.hum.z_score.jsonl"

    try:
        suspect_attack_wm_list = read_jsonl(suspect_attack_wm_file)
        suspect_attack_hum_list = read_jsonl(suspect_attack_hum_file)
    except:
        print(f"Error: Could not load main files for {target_lang}")
        return None, None

    if len(suspect_attack_wm_list) != num_samples or len(suspect_attack_hum_list) != num_samples:
        print(f"Error: Expected {num_samples} samples, got {len(suspect_attack_wm_list)} wm, {len(suspect_attack_hum_list)} hm")
        return None, None

    # Load intermediate language files
    intermediate_wm_data = {}
    intermediate_hm_data = {}

    for lang in ORG_LANGS:
        if lang == target_lang:
            continue

        # Load watermark files
        wm_file = f"{base_dir}/mc4.{target_lang}-{lang}-back.mod.z_score.jsonl"
        hm_file = f"{base_dir}/mc4.{target_lang}-{lang}-back.hum.z_score.jsonl"
        val_file = f"{base_dir}/mc4.{target_lang}-{lang}-back.val.z_score.jsonl"

        try:
            wm_list = read_jsonl(wm_file)
            hm_list = read_jsonl(hm_file)

            if len(wm_list) == len(hm_list) == num_samples:
                # Get validation normalization
                avg_val_zscore = get_avg_zscore(val_file)

                intermediate_wm_data[lang] = {
                    'data': wm_list,
                    'avg_val': avg_val_zscore
                }
                intermediate_hm_data[lang] = {
                    'data': hm_list,
                    'avg_val': avg_val_zscore
                }
        except:
            continue  # Skip languages that don't have files

    print(f"Loaded {len(intermediate_wm_data)} intermediate languages for {target_lang}")

    # Process each text individually
    wm_best_scores = []
    hm_best_scores = []

    for text_idx in range(num_samples):
        # For watermarked text
        wm_scores = []

        # Check suspect attack (en -> target_lang direct)
        suspect_wm_score = get_zscore_for_text(suspect_attack_wm_list, text_idx)
        wm_scores.append(suspect_wm_score)

        # Check all intermediate languages
        for lang in intermediate_wm_data:
            raw_score = get_zscore_for_text(intermediate_wm_data[lang]['data'], text_idx)
            normalized_score = raw_score - intermediate_wm_data[lang]['avg_val']
            wm_scores.append(normalized_score)

        # Best score for this watermarked text
        best_wm_score = max(wm_scores) if wm_scores else 0
        wm_best_scores.append(best_wm_score)

        # For human text
        hm_scores = []

        # Check suspect attack (en -> target_lang direct)
        suspect_hm_score = get_zscore_for_text(suspect_attack_hum_list, text_idx)
        hm_scores.append(suspect_hm_score)

        # Check all intermediate languages (raw scores for human)
        for lang in intermediate_hm_data:
            raw_score = get_zscore_for_text(intermediate_hm_data[lang]['data'], text_idx)
            hm_scores.append(raw_score)

        # Best score for this human text
        best_hm_score = max(hm_scores) if hm_scores else 0
        hm_best_scores.append(best_hm_score)

    return wm_best_scores, hm_best_scores

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
    wm_best_scores, hm_best_scores = steam_per_text_simple(args.base_dir, args.tgt_lang)

    if wm_best_scores is None or hm_best_scores is None:
        print("Failed to process files")
        return

    print(f"Processed {len(wm_best_scores)} watermarked and {len(hm_best_scores)} human texts")

    # ROC Analysis
    y_true = [0] * len(hm_best_scores) + [1] * len(wm_best_scores)  # 0=human, 1=watermarked
    y_scores = hm_best_scores + wm_best_scores

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    print(f"""STEAM Simple Per-Text Results for {args.tgt_lang}:

AUC: {auc:.3f}

TPR@FPR=0.1: {tpr_at_fpr(fpr, tpr, 0.1):.3f}
TPR@FPR=0.01: {tpr_at_fpr(fpr, tpr, 0.01):.3f}

F1@FPR=0.1: {f1_at_fpr(y_true, y_scores, 0.1):.3f}
F1@FPR=0.01: {f1_at_fpr(y_true, y_scores, 0.01):.3f}

Average human best score: {np.mean(hm_best_scores):.3f}
Average watermark best score: {np.mean(wm_best_scores):.3f}
""")

    # Save ROC curve if requested
    if args.roc_curve:
        with open(args.roc_curve, "w") as f:
            f.write(f"FPR\tTPR\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]:.3f}\t{tpr[i]:.3f}\n")

        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='red', linewidth=2)
        tpr_0_1 = tpr_at_fpr(fpr, tpr, 0.1)
        tpr_0_01 = tpr_at_fpr(fpr, tpr, 0.01)
        plt.scatter(0.1, tpr_0_1, color='red', s=40, edgecolors='black', label=f'TPR@0.1: {tpr_0_1:.3f}')
        plt.scatter(0.01, tpr_0_01, color='blue', s=40, edgecolors='black', label=f'TPR@0.01: {tpr_0_01:.3f}')
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc='lower right', fontsize=8, frameon=True)
        plt.grid()
        roc_curve_file = args.roc_curve.split(".txt")[0]
        plt.savefig(f"{roc_curve_file}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple STEAM per-text evaluation")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for z-score files")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--roc_curve", type=str, help="ROC curve output file")

    args = parser.parse_args()
    main(args)