import argparse
import matplotlib.pyplot as plt
import json
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import numpy as np

from utils import read_jsonl


ORG_LANGS = [
    "en",  # English
    # High-resource languages
        # "fr" # French
    "de", # German
    # "it" # Italian
    "es", # Spanish
    # "pt" # Portuguese
    # Medium-resource languages
    # "pl" # Polish
    # "nl" # Dutch
    "ru", # Russian
    "hi", # Hindi
    # "ko" # Korean
    # "ja" # Japanese
    # Low-resource languages
    "bn", # Bengali
    "fa", # Persian
    # "vi" # Vietnamese
    # "iw" # Hebrew
    # "uk" # Ukrainian
    # "ta" # Tamil
]


def extract_zscores(_list):
    """Extract z-scores from list of dictionaries."""
    return [_["z_score"] if _["z_score"] is not None else 0 for _ in _list]


def get_avg_zscore(validation_file):
    """Compute average z-score from validation file."""
    validation_list = read_jsonl(validation_file)
    zscores = extract_zscores(validation_list)
    return sum(zscores) / len(zscores) if zscores else 0


def tpr_at_fpr(fpr, tpr, fpr_target):
    """Interpolate TPR at specific FPR."""
    fpr_tpr_interpolation = interpolate.interp1d(fpr, tpr, kind="linear")
    return float(fpr_tpr_interpolation(fpr_target))


def fpr_at_tpr(fpr, tpr, tpr_target):
    """
    Interpolate FPR at specific TPR.
    This is the NEW metric your supervisor wants!
    """
    # ROC curve gives us (fpr, tpr) pairs
    # We need to invert: given tpr_target, find fpr
    
    # Create interpolation function (note: tpr is x-axis now!)
    tpr_fpr_interpolation = interpolate.interp1d(tpr, fpr, kind="linear", 
                                                  bounds_error=False, 
                                                  fill_value=(fpr[0], fpr[-1]))
    return float(tpr_fpr_interpolation(tpr_target))



def f1_at_fpr(y_true, y_scores, fpr_target):
    """Calculate F1 score at specific FPR."""
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

    return float(f1)


def load_candidate_scores(args, tgt_lang, num_samples):
    """
    Load all back-translation scores for all languages.
    Returns dictionaries mapping language -> list of z-scores.
    """
    candidate_hum_zscore = {}
    candidate_wm_zscore = {}
    
    for lang in ORG_LANGS:
        if lang == tgt_lang:
            continue
        
        hum_zscore_file = args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.hum.z_score.jsonl"
        wm_zscore_file = args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.mod.z_score.jsonl"
        
        try:
            hum_list = read_jsonl(hum_zscore_file)
            wm_list = read_jsonl(wm_zscore_file)
            
            if len(hum_list) != num_samples or len(wm_list) != num_samples:
                print(f"Warning: Expected {num_samples} samples for {lang}, got {len(hum_list)}/{len(wm_list)}")
                continue
                
            candidate_hum_zscore[lang] = extract_zscores(hum_list)
            candidate_wm_zscore[lang] = extract_zscores(wm_list)
        except FileNotFoundError:
            print(f"Warning: Files not found for language {lang}")
            continue
    
    return candidate_hum_zscore, candidate_wm_zscore


def compute_steam_max_scores(args, tgt_lang, num_samples, 
                             candidate_hum_zscore, candidate_wm_zscore,
                             suspect_hum_zscores, suspect_wm_zscores):
    """
    Compute maximum normalized z-scores across all back-translations (STEAM method).
    Returns lists of maximum scores for human and watermarked texts.
    """
    steam_hum_max_scores = []
    steam_wm_max_scores = []
    correct_hum_lang = 0
    correct_wm_lang = 0
    true_lang = "en"
    
    for i in range(num_samples):
        # For human texts
        max_hum_score = suspect_hum_zscores[i]  # Include original suspect text
        best_hum_lang = tgt_lang
        
        for lang in ORG_LANGS:
            if lang == tgt_lang or lang not in candidate_hum_zscore:
                continue
            
            # Get validation average for normalization
            avg_val_zscore = get_avg_zscore(
                args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.val.z_score.jsonl"
            )
            
            # Normalize score
            normalized_score = candidate_hum_zscore[lang][i] - avg_val_zscore
            
            if normalized_score > max_hum_score:
                max_hum_score = normalized_score
                best_hum_lang = lang
        
        steam_hum_max_scores.append(max_hum_score)
        if best_hum_lang == true_lang:
            correct_hum_lang += 1
        
        # For watermarked texts
        max_wm_score = suspect_wm_zscores[i]  # Include original suspect text
        best_wm_lang = tgt_lang
        
        for lang in ORG_LANGS:
            if lang == tgt_lang or lang not in candidate_wm_zscore:
                continue
            
            # Get validation average for normalization
            avg_val_zscore = get_avg_zscore(
                args.base_wm_dir + f"/mc4.{tgt_lang}-{lang}-back.val.z_score.jsonl"
            )
            
            # Normalize score
            normalized_score = candidate_wm_zscore[lang][i] - avg_val_zscore
            
            if normalized_score > max_wm_score:
                max_wm_score = normalized_score
                best_wm_lang = lang
        
        steam_wm_max_scores.append(max_wm_score)
        if best_wm_lang == true_lang:
            correct_wm_lang += 1
    
    return steam_hum_max_scores, steam_wm_max_scores, correct_wm_lang, correct_hum_lang


def run_standard_evaluation(args):
    """
    Run the standard STEAM evaluation with BOTH metrics:
    1. TPR@FPR (original)
    2. FPR@TPR (new - what supervisor wants)
    """
    num_samples = 500
    tgt_lang = args.tgt_lang
    true_lang = "en"
    
    # Load suspect attack scores
    suspect_attack_wm_file = args.base_wm_dir + f"/mc4.en-{tgt_lang}.mod.z_score.jsonl"
    suspect_attack_hum_file = args.base_wm_dir + f"/mc4.en-{tgt_lang}.hum.z_score.jsonl"
    
    suspect_attack_wm_list = read_jsonl(suspect_attack_wm_file)
    suspect_attack_hum_list = read_jsonl(suspect_attack_hum_file)
    
    if len(suspect_attack_wm_list) != num_samples or len(suspect_attack_hum_list) != num_samples:
        print("Error: The number of zscores in the suspect attack file is not correct.")
        return None
    
    suspect_attack_wm_zscore = extract_zscores(suspect_attack_wm_list)
    suspect_attack_hum_zscore = extract_zscores(suspect_attack_hum_list)
    
    # Load candidate scores
    candidate_hum_zscore, candidate_wm_zscore = load_candidate_scores(args, tgt_lang, num_samples)
    
    if not candidate_hum_zscore or not candidate_wm_zscore:
        print("Error: Could not load candidate scores.")
        return None
    
    # Compute STEAM maximum scores
    maximum_hum_zscore, maximum_wm_zscore, correct_wm_lang, correct_hum_lang = compute_steam_max_scores(
        args, tgt_lang, num_samples,
        candidate_hum_zscore, candidate_wm_zscore,
        suspect_attack_hum_zscore, suspect_attack_wm_zscore
    )
    
    # Prepare labels and scores
    hm_true = [0 for _ in range(num_samples)]
    wm_true = [1 for _ in range(num_samples)]

    y_true = hm_true + wm_true
    y_scores = maximum_hum_zscore + maximum_wm_zscore

    # Calculate AUC and ROC curve
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Original metrics
    tpr_at_fpr_10 = tpr_at_fpr(fpr, tpr, 0.1)
    tpr_at_fpr_01 = tpr_at_fpr(fpr, tpr, 0.01)
    f1_at_fpr_10 = f1_at_fpr(y_true, y_scores, 0.1)
    f1_at_fpr_01 = f1_at_fpr(y_true, y_scores, 0.01)

    print(f"AUC: {auc:.3f}")
    # print(f"TPR@FPR=10%: {tpr_at_fpr_10:.3f}")
    # print(f"TPR@FPR=1%:  {tpr_at_fpr_01:.3f}")
    # print(f"F1@FPR=10%:  {f1_at_fpr_10:.3f}")
    # print(f"F1@FPR=1%:   {f1_at_fpr_01:.3f}")

    # New metrics
    # fpr_at_tpr_01_direct, threshold_01, actual_tpr_01 = compute_fpr_at_tpr_directly(y_true, y_scores, 0.01)
    # fpr_at_tpr_05_direct, threshold_05, actual_tpr_05 = compute_fpr_at_tpr_directly(y_true, y_scores, 0.05)
    fpr_at_tpr_99_interp = fpr_at_tpr(fpr, tpr, 0.99)
    fpr_at_tpr_95_interp = fpr_at_tpr(fpr, tpr, 0.95)
    fpr_at_tpr_90_interp = fpr_at_tpr(fpr, tpr, 0.90)

    # print(f"FPR@TPR=1%:  {fpr_at_tpr_01_direct:.4f} (threshold: {threshold_01:.3f}, actual TPR: {actual_tpr_01:.3f})")
    # print(f"FPR@TPR=5%:  {fpr_at_tpr_05_direct:.4f} (threshold: {threshold_05:.3f}, actual TPR: {actual_tpr_05:.3f})")
    print(f"FPR@TPR=99%:  {fpr_at_tpr_99_interp:.4f}")
    print(f"FPR@TPR=95%:  {fpr_at_tpr_95_interp:.4f}")
    print(f"FPR@TPR=90%: {fpr_at_tpr_90_interp:.4f}")


    return {
        'auc': float(auc),
        'original_metrics': {
            'tpr_at_fpr_0.1': float(tpr_at_fpr_10),
            'tpr_at_fpr_0.01': float(tpr_at_fpr_01),
            'f1_at_fpr_0.1': float(f1_at_fpr_10),
            'f1_at_fpr_0.01': float(f1_at_fpr_01)
        },
        'new_metrics': {
            'fpr_at_tpr_0.01_interp': float(fpr_at_tpr_99_interp),
            'fpr_at_tpr_0.05_interp': float(fpr_at_tpr_95_interp),
        },
        'lang_detection': {
            'accuracy_wm': float(correct_wm_lang / num_samples),
            'accuracy_hum': float(correct_hum_lang / num_samples)
        }
    }

def main(args):
    """
    Main function with both metric systems.
    """
    # Run standard evaluation with both metrics
    standard_results = run_standard_evaluation(args)

    if standard_results is None:
        return
    
    # Save all results
    if args.output_json:
        all_results = {
            'target_language': args.tgt_lang,
            'base_directory': args.base_wm_dir,
            'standard_evaluation': standard_results
        }
        
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STEAM Evaluation with Dual Metrics (TPR@FPR and FPR@TPR)"
    )
    parser.add_argument(
        "--base_wm_dir", 
        type=str, 
        required=True, 
        help="Base directory for watermark files"
    )
    parser.add_argument(
        "--tgt_lang", 
        type=str, 
        required=True, 
        help="Target language (e.g., 'fr', 'de', 'ta')"
    )
    parser.add_argument(
        "--roc_curve", 
        type=str, 
        default=None, 
        help="Output file for ROC curve data"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file for all results"
    )
    
    args = parser.parse_args()
    
    if args.output_json is None:
        args.output_json = f"steam_dual_metrics_{args.tgt_lang}.json"
    
    main(args)