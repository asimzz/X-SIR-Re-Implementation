import argparse
import matplotlib.pyplot as plt

from utils import read_jsonl
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


ORG_LANGS = [
    "en", # English
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
    "ja", # Japanese
    # Low-resource languages
    "bn", # Bengali
    "fa", # Persian
    # "vi" # Vietnamese
    "iw", # Hebrew
    # "uk" # Ukrainian
    # "ta" # Tamil
    ]


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


def evaluate_length_category(base_dir, tgt_lang, length_category, num_samples=500):
    true_lang = "en"

    # Load and organize data by prompt to align samples correctly
    candidate_hum_data = {}
    candidate_wm_data = {}

    for lang in ORG_LANGS:
        if lang == tgt_lang:
            continue
        hum_zscore_file = f"{base_dir}/mc4.{tgt_lang}-{lang}-back.hum.z_score.{length_category}.z_score.jsonl"
        wm_zscore_file = f"{base_dir}/mc4.{tgt_lang}-{lang}-back.mod.z_score.{length_category}.z_score.jsonl"

        try:
            hum_list = read_jsonl(hum_zscore_file)
            wm_list = read_jsonl(wm_zscore_file)
        except FileNotFoundError:
            print(f"Warning: Files not found for {lang}, skipping...")
            continue

        if len(hum_list) != len(wm_list):
            print(f"The number of zscores in the human and watermark files are not the same for {lang}.")
            continue

        # Organize by prompt to ensure alignment
        hum_by_prompt = {item['prompt']: item for item in hum_list}
        wm_by_prompt = {item['prompt']: item for item in wm_list}

        candidate_hum_data[lang] = hum_by_prompt
        candidate_wm_data[lang] = wm_by_prompt

    suspect_attack_wm_file = f"{base_dir}/mc4.en-{tgt_lang}.mod.z_score.{length_category}.z_score.jsonl"
    suspect_attack_hum_file = f"{base_dir}/mc4.en-{tgt_lang}.hum.z_score.{length_category}.z_score.jsonl"

    try:
        suspect_attack_wm_list = read_jsonl(suspect_attack_wm_file)
        suspect_attack_hum_list = read_jsonl(suspect_attack_hum_file)
    except FileNotFoundError:
        print(f"Suspect attack files not found for {length_category}.")
        return None

    # Organize suspect attack data by prompt
    suspect_wm_by_prompt = {item['prompt']: item for item in suspect_attack_wm_list}
    suspect_hum_by_prompt = {item['prompt']: item for item in suspect_attack_hum_list}

    # Get common prompts across all data to ensure proper alignment
    common_prompts = set(suspect_wm_by_prompt.keys()) & set(suspect_hum_by_prompt.keys())

    # Filter to prompts that exist in at least one candidate language
    valid_prompts = []
    for prompt in common_prompts:
        has_candidate_data = False
        for lang in candidate_hum_data.keys():
            if prompt in candidate_hum_data[lang] and prompt in candidate_wm_data[lang]:
                has_candidate_data = True
                break
        if has_candidate_data:
            valid_prompts.append(prompt)

    if len(valid_prompts) == 0:
        print(f"No valid prompts found for {length_category}.")
        return None

    print(f"Using {len(valid_prompts)} aligned samples for {length_category} evaluation.")

    # Load validation data organized by prompt
    val_avg_by_lang = {}
    for lang in candidate_hum_data.keys():
        val_file = f"{base_dir}/mc4.{tgt_lang}-{lang}-back.val.z_score.{length_category}.z_score.jsonl"
        try:
            avg_val_zscore = get_avg_zscore(val_file)
        except FileNotFoundError:
            avg_val_zscore = 0
        val_avg_by_lang[lang] = avg_val_zscore

    maximum_hum_zscore = []
    maximum_wm_zscore = []
    correct_hum_lang = 0
    correct_wm_lang = 0

    for prompt in valid_prompts:
        max_hum_score = float('-inf')
        max_wm_score = float('-inf')
        best_hum_lang = None
        best_wm_lang = None

        # Check all candidate languages for this prompt
        for lang in candidate_hum_data.keys():
            if prompt not in candidate_hum_data[lang] or prompt not in candidate_wm_data[lang]:
                continue

            avg_val_zscore = val_avg_by_lang[lang]

            hum_zscore = candidate_hum_data[lang][prompt]['z_score']
            wm_zscore = candidate_wm_data[lang][prompt]['z_score']

            # Handle None z_scores
            if hum_zscore is None:
                hum_zscore = 0
            if wm_zscore is None:
                wm_zscore = 0

            hum_score = hum_zscore - avg_val_zscore
            wm_score = wm_zscore - avg_val_zscore

            if hum_score > max_hum_score:
                max_hum_score = hum_score
                best_hum_lang = lang
            if wm_score > max_wm_score:
                max_wm_score = wm_score
                best_wm_lang = lang

        # Add suspect attack scores
        suspect_wm_zscore = suspect_wm_by_prompt[prompt]['z_score']
        suspect_hum_zscore = suspect_hum_by_prompt[prompt]['z_score']

        if suspect_wm_zscore is None:
            suspect_wm_zscore = 0
        if suspect_hum_zscore is None:
            suspect_hum_zscore = 0

        max_wm_score = max(max_wm_score, suspect_wm_zscore)
        max_hum_score = max(max_hum_score, suspect_hum_zscore)

        maximum_hum_zscore.append(max_hum_score)
        maximum_wm_zscore.append(max_wm_score)

        if best_hum_lang == true_lang:
            correct_hum_lang += 1
        if best_wm_lang == true_lang:
            correct_wm_lang += 1

    actual_samples = len(valid_prompts)
    accuracy = (correct_wm_lang / actual_samples) * 100

    hm_true = [0 for _ in maximum_hum_zscore]
    wm_true = [1 for _ in maximum_wm_zscore]

    y_true = hm_true + wm_true
    y_scores = maximum_hum_zscore + maximum_wm_zscore

    if len(set(y_true)) > 1 and len(y_scores) > 0:  # Check if we have both classes and data
        auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        tpr_01 = tpr_at_fpr(fpr, tpr, 0.1)
        tpr_001 = tpr_at_fpr(fpr, tpr, 0.01)
        f1_01 = f1_at_fpr(y_true, y_scores, 0.1)
        f1_001 = f1_at_fpr(y_true, y_scores, 0.01)
    else:
        auc = 0.0
        tpr_01 = 0.0
        tpr_001 = 0.0
        f1_01 = 0.0
        f1_001 = 0.0
        fpr = []
        tpr = []

    return {
        'accuracy': accuracy,
        'correct_wm_lang': correct_wm_lang,
        'auc': auc,
        'tpr_01': tpr_01,
        'tpr_001': tpr_001,
        'f1_01': f1_01,
        'f1_001': f1_001,
        'fpr': fpr,
        'tpr': tpr,
        'num_samples': actual_samples
    }


def main(args):
    num_samples = 500
    tgt_lang = args.tgt_lang
    length_categories = ['short', 'medium', 'long']

    results = {}

    for length_category in length_categories:
        print(f"\n=== Evaluating {length_category.upper()} texts ===")
        result = evaluate_length_category(args.base_wm_dir, tgt_lang, length_category, num_samples)

        if result is None:
            print(f"Skipping {length_category} due to missing files or errors.")
            continue

        results[length_category] = result

        print(f"Correct watermark language detection count: {result['correct_wm_lang']}/{result['num_samples']}")
        print(f"Accuracy: {result['accuracy']:.1f}%")
        print(f"AUC: {result['auc']:.3f}")
        print(f"TPR@FPR=0.1: {result['tpr_01']:.3f}")
        print(f"TPR@FPR=0.01: {result['tpr_001']:.3f}")
        print(f"F1@FPR=0.1: {result['f1_01']:.3f}")
        print(f"F1@FPR=0.01: {result['f1_001']:.3f}")

    # Summary comparison
    if results:
        print(f"\n=== SUMMARY COMPARISON ===")
        print(f"{'Category':<10} {'Accuracy':<10} {'AUC':<8} {'TPR@0.1':<8} {'F1@0.1':<8}")
        print("-" * 50)
        for category, result in results.items():
            print(f"{category:<10} {result['accuracy']:>7.1f}%  {result['auc']:>6.3f}  {result['tpr_01']:>6.3f}  {result['f1_01']:>6.3f}")

    # Generate ROC curves if requested
    if args.roc_curve and results:
        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue', 'green']

        for i, (category, result) in enumerate(results.items()):
            if len(result['fpr']) > 0 and len(result['tpr']) > 0:
                plt.plot(result['fpr'], result['tpr'],
                        label=f"{category.capitalize()} (AUC = {result['auc']:.3f})",
                        color=colors[i], linewidth=2)

        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curves by Text Length")
        plt.legend(loc='lower right', fontsize=10, frameon=True)
        plt.grid(True, alpha=0.3)
        plt.savefig(args.roc_curve, dpi=300, bbox_inches='tight')

        # Save data to text file
        with open(args.roc_curve.replace('.png', '.txt'), "w") as f:
            for category, result in results.items():
                f.write(f"# {category.upper()}\n")
                f.write(f"FPR\tTPR\n")
                for j in range(len(result['fpr'])):
                    f.write(f"{result['fpr'][j]:.3f}\t{result['tpr'][j]:.3f}\n")
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate watermark detection across text lengths")
    parser.add_argument(
        "--base_wm_dir", type=str, required=True, help="Base directory for watermark files"
    )
    parser.add_argument("--roc_curve", type=str, default=None, help="ROC curve output file (.png)")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")

    args = parser.parse_args()
    main(args)