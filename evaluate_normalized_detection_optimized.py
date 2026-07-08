import argparse
import matplotlib.pyplot as plt
import csv
import ast
from pathlib import Path

from utils import read_jsonl
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import json


def get_optimized_pivot_languages(target_lang, pivot_csv="pivot_language_extraction/max_distance_hierarchical_clean.csv"):
    """
    Get optimized pivot languages for the target language from CSV file.

    Args:
        target_lang: Target language (2-letter code like 'fr')
        pivot_csv: Path to the CSV file with optimized pivot languages

    Returns:
        List of 2-letter pivot language codes
    """
    if not Path(pivot_csv).exists():
        raise FileNotFoundError(f"Optimized pivot languages file not found: {pivot_csv}")

    with open(pivot_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row[0] == target_lang:
                # Parse the list string and convert 3-letter to 2-letter codes
                lang_list_3letter = ast.literal_eval(row[1])
                lang_list_2letter = [convert_3letter_to_2letter(lang) for lang in lang_list_3letter]
                return lang_list_2letter

    raise ValueError(f"No optimized pivot languages found for target language: {target_lang}")


def convert_3letter_to_2letter(lang_3):
    """Convert 3-letter language codes to 2-letter codes"""
    mapping = {
        "fin": "fi", "hun": "hu", "tur": "tr", "uig": "ug",  # Uyghur
        "spa": "es", "fra": "fr", "eng": "en", "deu": "de",
        "ita": "it", "por": "pt", "pol": "pl", "nld": "nl",
        "rus": "ru", "hin": "hi", "kor": "ko", "jpn": "ja",
        "ben": "bn", "fas": "fa", "vie": "vi", "heb": "iw",
        "ukr": "uk", "tam": "ta"
    }
    return mapping.get(lang_3, lang_3)  # fallback to original if not found


def extract_zscores(_list):
    return [_["z_score"] if _["z_score"] is not None else 0 for _ in _list]


def get_avg_zscore(validation_file):
    validation_list = read_jsonl(validation_file)
    zscores = extract_zscores(validation_list)
    return sum(zscores) / len(zscores) if zscores else 0


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

    # Get optimized pivot languages for the target language
    tgt_lang = args.tgt_lang
    try:
        optimized_pivot_langs = get_optimized_pivot_languages(tgt_lang, args.pivot_csv)
        print(f"🧬 Using OPTIMIZED pivot languages for {tgt_lang}: {optimized_pivot_langs}")
    except Exception as e:
        print(f"❌ Error loading optimized pivot languages: {e}")
        print(f"Please ensure {args.pivot_csv} exists and contains data for {tgt_lang}")
        return

    wm_zscore = []
    candidate_hum_zscore = {}
    candidate_wm_zscore = {}

    # Load z-scores for OPTIMIZED pivot languages only
    for lang in optimized_pivot_langs:
        if lang == tgt_lang:
            continue

        hum_zscore_file = args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.hum.z_score.jsonl"
        wm_zscore_file = args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.mod.z_score.jsonl"

        # Check if files exist
        if not Path(hum_zscore_file).exists() or not Path(wm_zscore_file).exists():
            print(f"⚠️  Warning: Missing files for pivot language {lang}")
            print(f"   Expected: {hum_zscore_file}")
            print(f"             {wm_zscore_file}")
            continue

        hum_list = read_jsonl(hum_zscore_file)
        wm_list = read_jsonl(wm_zscore_file)

        if len(hum_list) != len(wm_list):
            print(f"❌ Error: Mismatched z-score counts for language {lang}")
            print(f"   Human: {len(hum_list)}, Watermark: {len(wm_list)}")
            continue

        hum_zscore = extract_zscores(hum_list)
        wm_zscore = extract_zscores(wm_list)
        candidate_hum_zscore[lang] = hum_zscore
        candidate_wm_zscore[lang] = wm_zscore

    if not candidate_hum_zscore:
        print("❌ No valid pivot language z-score files found!")
        return

    # Load suspect attack files (translated target language)
    suspect_attack_wm_file = args.base_wm_dir + f"/mc4.en-{tgt_lang}.mod.z_score.jsonl"
    suspect_attack_hum_file = args.base_wm_dir + f"/mc4.en-{tgt_lang}.hum.z_score.jsonl"

    if not Path(suspect_attack_wm_file).exists() or not Path(suspect_attack_hum_file).exists():
        print(f"❌ Missing suspect attack files:")
        print(f"   {suspect_attack_wm_file}")
        print(f"   {suspect_attack_hum_file}")
        return

    suspect_attack_wm_list = read_jsonl(suspect_attack_wm_file)
    suspect_attack_hum_list = read_jsonl(suspect_attack_hum_file)

    if len(suspect_attack_wm_list) != num_samples or len(suspect_attack_hum_list) != num_samples:
        print(f"❌ Wrong number of samples in suspect attack files:")
        print(f"   Expected: {num_samples}")
        print(f"   WM: {len(suspect_attack_wm_list)}, Hum: {len(suspect_attack_hum_list)}")
        return

    suspect_attack_wm_zscore = extract_zscores(suspect_attack_wm_list)
    suspect_attack_hum_zscore = extract_zscores(suspect_attack_hum_list)

    # STEAM evaluation using OPTIMIZED pivot languages
    maximum_hum_zscore = []
    maximum_wm_zscore = []
    correct_hum_lang = 0
    correct_wm_lang = 0

    print(f"\n🔍 Running STEAM evaluation with {len(optimized_pivot_langs)} optimized pivot languages...")

    for i in range(num_samples):
        max_hum_score = float('-inf')
        max_wm_score = float('-inf')
        best_hum_lang = None
        best_wm_lang = None

        # Find maximum z-score across OPTIMIZED pivot languages
        for lang in optimized_pivot_langs:
            if lang == tgt_lang or lang not in candidate_hum_zscore:
                continue

            val_file = args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.val.z_score.jsonl"
            if not Path(val_file).exists():
                print(f"⚠️  Warning: Missing validation file for {lang}: {val_file}")
                continue

            avg_val_zscore = get_avg_zscore(val_file)
            hum_score = candidate_hum_zscore[lang][i] - avg_val_zscore
            wm_score = candidate_wm_zscore[lang][i] - avg_val_zscore

            if hum_score > max_hum_score:
                max_hum_score = hum_score
                best_hum_lang = lang
            if wm_score > max_wm_score:
                max_wm_score = wm_score
                best_wm_lang = lang

        # Include direct translation scores
        max_wm_score = max(max_wm_score, suspect_attack_wm_zscore[i])
        max_hum_score = max(max_hum_score, suspect_attack_hum_zscore[i])

        maximum_hum_zscore.append(max_hum_score)
        maximum_wm_zscore.append(max_wm_score)

        if best_hum_lang == true_lang:
            correct_hum_lang += 1
        if best_wm_lang == true_lang:
            correct_wm_lang += 1


    accuracy = (correct_wm_lang / num_samples) * 100

    hm_true = [0 for _ in range(num_samples)]
    wm_true = [1 for _ in range(num_samples)]

    y_true = hm_true + wm_true
    y_scores = maximum_hum_zscore + maximum_wm_zscore

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    print(f"AUC: {auc:.3f}")
    print(f"TPR@FPR=0.1: {tpr_at_fpr(fpr, tpr, 0.1):.3f}")
    print(f"TPR@FPR=0.01: {tpr_at_fpr(fpr, tpr, 0.01):.3f}")
    print(f"F1@FPR=0.1: {f1_at_fpr(y_true, y_scores, 0.1):.3f}")
    print(f"F1@FPR=0.01: {f1_at_fpr(y_true, y_scores, 0.01):.3f}")

    if args.roc_curve:
        with open(args.roc_curve, "w") as f:
            f.write(f"FPR\tTPR\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]:.3f}\t{tpr[i]:.3f}\n")

        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f'Optimized AUC = {auc:.3f}', color='red', linewidth=2)
        tpr_0_1 = tpr_at_fpr(fpr, tpr, 0.1)
        tpr_0_01 = tpr_at_fpr(fpr, tpr, 0.01)
        plt.scatter(0.1, tpr_0_1, color='red', s=40, edgecolors='black', label=f'TPR@0.1: {tpr_0_1:.3f}')
        plt.scatter(0.01, tpr_0_01, color='blue', s=40, edgecolors='black', label=f'TPR@0.01: {tpr_0_01:.3f}')
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"STEAM with Optimized Pivots ({tgt_lang})")
        plt.legend(loc='lower right', fontsize=8, frameon=True)
        plt.grid()
        roc_curve_file = args.roc_curve.split(".txt")[0]
        plt.savefig(f"{roc_curve_file}_optimized.png", dpi=300, bbox_inches='tight')
        print(f"📈 ROC curve saved to: {roc_curve_file}_optimized.png")

    print(f"\n🎉 Optimized STEAM evaluation complete!")
    print(f"🧬 Used genetic distance-optimized pivot selection instead of fixed languages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEAM evaluation with optimized pivot languages")
    parser.add_argument(
        "--base_wm_dir", type=str, required=True,
        help="Base directory for watermark files"
    )
    parser.add_argument(
        "--roc_curve", type=str, default=None,
        help="ROC curve output file"
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
        help="Target language (2-letter code, e.g., 'fr')"
    )
    parser.add_argument(
        "--pivot_csv", type=str,
        default="pivot_language_extraction/max_distance_hierarchical_clean.csv",
        help="Path to CSV file with optimized pivot languages"
    )

    args = parser.parse_args()
    main(args)