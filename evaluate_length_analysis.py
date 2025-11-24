import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict

from utils import read_jsonl
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


ORG_LANGS = [
    "en", # English
    "de", # German
    "es", # Spanish
    "ru", # Russian
    "hi", # Hindi
    "ja", # Japanese
    "bn", # Bengali
    "fa", # Persian
    "iw", # Hebrew
]


def get_text_length_from_tokenizer(text, tokenizer_name="CohereForAI/aya-23-8B"):
    """Get text length in tokens using the specified tokenizer"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return len(tokenizer.encode(text, add_special_tokens=False))


def classify_text_length_by_percentiles(token_length, percentiles):
    """Classify texts into length categories based on percentile thresholds"""
    p33, p67 = percentiles
    if token_length <= p33:
        return "short"
    elif token_length <= p67:
        return "medium"
    else:
        return "long"


def calculate_length_percentiles(text_items, tokenizer):
    """Calculate 33rd and 67th percentiles of text lengths to split into thirds"""
    token_lengths = []
    for item in text_items:
        response = item["response"]
        token_length = len(tokenizer.encode(response, add_special_tokens=False))
        token_lengths.append(token_length)

    token_lengths.sort()
    n = len(token_lengths)
    if n == 0:
        return (0, 0)

    p33_idx = int(n * 0.33)
    p67_idx = int(n * 0.67)

    return (token_lengths[p33_idx], token_lengths[p67_idx])


def extract_zscores(_list):
    return [_["z_score"] if _["z_score"] is not None else 0 for _ in _list]


def get_avg_zscore(validation_file):
    validation_list = read_jsonl(validation_file)
    zscores = extract_zscores(validation_list)
    return sum(zscores) / len(zscores) if zscores else 0


def tpr_at_fpr(fpr, tpr, fpr_target):
    if len(fpr) == 0 or len(tpr) == 0:
        return 0.0
    fpr_tpr_interpolation = interpolate.interp1d(fpr, tpr, kind="linear")
    return float(fpr_tpr_interpolation(fpr_target))


def f1_at_fpr(y_true, y_scores, fpr_target):
    if len(set(y_true)) <= 1 or len(y_scores) == 0:
        return 0.0

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
    if precision_at_threshold + recall_at_threshold == 0:
        return 0.0

    f1 = (
        2
        * (precision_at_threshold * recall_at_threshold)
        / (precision_at_threshold + recall_at_threshold)
    )

    return float(f1)


def analyze_length_distribution(base_dir, tgt_lang):
    """Analyze the distribution of text lengths in the dataset using percentile-based binning"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-23-8B", trust_remote_code=True)

    # Load suspect attack data (this represents our main evaluation set)
    suspect_attack_hum_file = f"{base_dir}/mc4.en-{tgt_lang}.hum.z_score.jsonl"
    suspect_attack_wm_file = f"{base_dir}/mc4.en-{tgt_lang}.mod.z_score.jsonl"

    try:
        suspect_attack_hum_list = read_jsonl(suspect_attack_hum_file)
        suspect_attack_wm_list = read_jsonl(suspect_attack_wm_file)
    except FileNotFoundError:
        print(f"Suspect attack files not found for {tgt_lang}.")
        return {}, {}, (0, 0)

    # Calculate percentiles from the human text data to ensure equal splits
    percentiles = calculate_length_percentiles(suspect_attack_hum_list, tokenizer)
    p33, p67 = percentiles

    length_counts = {"short": 0, "medium": 0, "long": 0}
    length_examples = defaultdict(list)

    for item in suspect_attack_hum_list:
        response = item["response"]
        token_length = len(tokenizer.encode(response, add_special_tokens=False))
        length_category = classify_text_length_by_percentiles(token_length, percentiles)
        length_counts[length_category] += 1

        if len(length_examples[length_category]) < 3:  # Store a few examples
            length_examples[length_category].append(token_length)

    print(f"Length percentiles for {tgt_lang}: P33={p33}, P67={p67}")
    return length_counts, length_examples, percentiles


def evaluate_by_length_and_language(base_dir, tgt_lang, percentiles):
    """
    Evaluate watermark detection performance by both text length AND language
    Uses percentile-based binning to ensure equal thirds per language
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-23-8B", trust_remote_code=True)

    true_lang = "en"
    results = {}

    # Load suspect attack data and classify by length
    suspect_attack_hum_file = f"{base_dir}/mc4.en-{tgt_lang}.hum.z_score.jsonl"
    suspect_attack_wm_file = f"{base_dir}/mc4.en-{tgt_lang}.mod.z_score.jsonl"

    try:
        suspect_attack_hum_list = read_jsonl(suspect_attack_hum_file)
        suspect_attack_wm_list = read_jsonl(suspect_attack_wm_file)
    except FileNotFoundError:
        print(f"Suspect attack files not found for {tgt_lang}.")
        return {}

    # Organize suspect attack data by prompt and classify by length
    suspect_hum_by_prompt = {item['prompt']: item for item in suspect_attack_hum_list}
    suspect_wm_by_prompt = {item['prompt']: item for item in suspect_attack_wm_list}

    # Classify all prompts by text length using percentile-based classification
    prompts_by_length = {"short": [], "medium": [], "long": []}
    for prompt, item in suspect_hum_by_prompt.items():
        if prompt in suspect_wm_by_prompt:
            response = item["response"]
            token_length = len(tokenizer.encode(response, add_special_tokens=False))
            length_category = classify_text_length_by_percentiles(token_length, percentiles)
            prompts_by_length[length_category].append(prompt)

    # Load candidate language data
    candidate_data = {}
    for lang in ORG_LANGS:
        if lang == tgt_lang:
            continue

        hum_file = f"{base_dir}/mc4.{tgt_lang}-{lang}-back.hum.z_score.jsonl"
        wm_file = f"{base_dir}/mc4.{tgt_lang}-{lang}-back.mod.z_score.jsonl"
        val_file = f"{base_dir}/mc4.{tgt_lang}-{lang}-back.val.z_score.jsonl"

        try:
            hum_list = read_jsonl(hum_file)
            wm_list = read_jsonl(wm_file)
            avg_val_zscore = get_avg_zscore(val_file)
        except FileNotFoundError:
            print(f"Warning: Files not found for {lang}, skipping...")
            continue

        hum_by_prompt = {item['prompt']: item for item in hum_list}
        wm_by_prompt = {item['prompt']: item for item in wm_list}

        candidate_data[lang] = {
            'hum': hum_by_prompt,
            'wm': wm_by_prompt,
            'val_avg': avg_val_zscore
        }

    # Evaluate each length category
    for length_category in ["short", "medium", "long"]:
        prompts_in_category = prompts_by_length[length_category]

        if len(prompts_in_category) == 0:
            print(f"No {length_category} texts found for {tgt_lang}")
            continue

        print(f"\n=== Analyzing {length_category.upper()} texts for {tgt_lang} ===")
        print(f"Number of {length_category} texts: {len(prompts_in_category)}")

        # Results for this length category
        length_results = {}

        # Evaluate detection performance for this length category
        maximum_hum_zscore = []
        maximum_wm_zscore = []
        correct_wm_lang = 0
        lang_detection_counts = defaultdict(int)

        for prompt in prompts_in_category:
            if prompt not in suspect_hum_by_prompt or prompt not in suspect_wm_by_prompt:
                continue

            max_hum_score = float('-inf')
            max_wm_score = float('-inf')
            best_wm_lang = None

            # Check all candidate languages
            for lang in candidate_data.keys():
                if (prompt not in candidate_data[lang]['hum'] or
                    prompt not in candidate_data[lang]['wm']):
                    continue

                hum_zscore = candidate_data[lang]['hum'][prompt]['z_score']
                wm_zscore = candidate_data[lang]['wm'][prompt]['z_score']
                val_avg = candidate_data[lang]['val_avg']

                if hum_zscore is None:
                    hum_zscore = 0
                if wm_zscore is None:
                    wm_zscore = 0

                hum_score = hum_zscore - val_avg
                wm_score = wm_zscore - val_avg

                if hum_score > max_hum_score:
                    max_hum_score = hum_score
                if wm_score > max_wm_score:
                    max_wm_score = wm_score
                    best_wm_lang = lang

            # Add suspect attack scores
            suspect_hum_zscore = suspect_hum_by_prompt[prompt]['z_score'] or 0
            suspect_wm_zscore = suspect_wm_by_prompt[prompt]['z_score'] or 0

            final_hum_score = max(max_hum_score, suspect_hum_zscore)
            final_wm_score = max(max_wm_score, suspect_wm_zscore)

            maximum_hum_zscore.append(final_hum_score)
            maximum_wm_zscore.append(final_wm_score)

            # Track language detection accuracy
            if best_wm_lang == true_lang:
                correct_wm_lang += 1
            if best_wm_lang:
                lang_detection_counts[best_wm_lang] += 1

        # Calculate metrics
        num_samples = len(maximum_hum_zscore)
        accuracy = (correct_wm_lang / num_samples * 100) if num_samples > 0 else 0

        # ROC analysis
        if len(maximum_hum_zscore) > 0 and len(maximum_wm_zscore) > 0:
            hm_true = [0] * len(maximum_hum_zscore)
            wm_true = [1] * len(maximum_wm_zscore)
            y_true = hm_true + wm_true
            y_scores = maximum_hum_zscore + maximum_wm_zscore

            if len(set(y_true)) > 1:
                auc = roc_auc_score(y_true, y_scores)
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                tpr_01 = tpr_at_fpr(fpr, tpr, 0.1)
                tpr_001 = tpr_at_fpr(fpr, tpr, 0.01)
                f1_01 = f1_at_fpr(y_true, y_scores, 0.1)
                f1_001 = f1_at_fpr(y_true, y_scores, 0.01)
            else:
                auc = tpr_01 = tpr_001 = f1_01 = f1_001 = 0.0
                fpr = tpr = []
        else:
            auc = tpr_01 = tpr_001 = f1_01 = f1_001 = 0.0
            fpr = tpr = []

        length_results = {
            'num_samples': num_samples,
            'accuracy': accuracy,
            'correct_detections': correct_wm_lang,
            'auc': auc,
            'tpr_01': tpr_01,
            'tpr_001': tpr_001,
            'f1_01': f1_01,
            'f1_001': f1_001,
            'fpr': fpr,
            'tpr': tpr,
            'lang_detection_counts': dict(lang_detection_counts)
        }

        results[length_category] = length_results

        # Print results for this length category
        print(f"Samples: {num_samples}")
        print(f"Language Detection Accuracy: {accuracy:.1f}% ({correct_wm_lang}/{num_samples})")
        print(f"AUC: {auc:.3f}")
        print(f"TPR@FPR=0.1: {tpr_01:.3f}")
        print(f"TPR@FPR=0.01: {tpr_001:.3f}")
        print(f"F1@FPR=0.1: {f1_01:.3f}")
        print(f"F1@FPR=0.01: {f1_001:.3f}")
        if lang_detection_counts:
            print(f"Language detection breakdown: {dict(lang_detection_counts)}")

    return results


def save_results_table(results, tgt_lang, output_csv=None):
    """Save results as a structured table - exactly what's shown in the charts"""

    # Create the main results table (same as before)
    table_data = []
    for length_category in ["short", "medium", "long"]:
        if length_category in results:
            r = results[length_category]
            table_data.append({
                'Language': tgt_lang,
                'Length_Category': length_category.capitalize(),
                'Num_Samples': r['num_samples'],
                'Accuracy_%': round(r['accuracy'], 1),
                'AUC': round(r['auc'], 3),
                'TPR_at_FPR_0.1': round(r['tpr_01'], 3),
                'TPR_at_FPR_0.01': round(r['tpr_001'], 3),
                'F1_at_FPR_0.1': round(r['f1_01'], 3),
                'F1_at_FPR_0.01': round(r['f1_001'], 3)
            })

    df = pd.DataFrame(table_data)

    # Print formatted table
    print(f"\n=== RESULTS TABLE FOR {tgt_lang.upper()} ===")
    print(df.to_string(index=False, float_format='%.3f'))

    # Create and print the chart-equivalent tables
    print(f"\n=== CHART DATA AS TABLES FOR {tgt_lang.upper()} ===")

    # Table 1: Sample Distribution (from bottom-left chart)
    print(f"\n📊 Sample Distribution by Length")
    print("-" * 40)
    sample_table = pd.DataFrame({
        'Length_Category': [r['Length_Category'] for r in table_data],
        'Number_of_Samples': [r['Num_Samples'] for r in table_data]
    })
    print(sample_table.to_string(index=False))

    # Table 2: AUC Scores (from bottom-right chart)
    print(f"\n🎯 AUC Scores by Text Length")
    print("-" * 40)
    auc_table = pd.DataFrame({
        'Length_Category': [r['Length_Category'] for r in table_data],
        'AUC_Score': [r['AUC'] for r in table_data]
    })
    print(auc_table.to_string(index=False))

    # Table 3: Detection Accuracy (from top-right chart)
    print(f"\n🎯 Detection Accuracy by Text Length")
    print("-" * 40)
    acc_table = pd.DataFrame({
        'Length_Category': [r['Length_Category'] for r in table_data],
        'Accuracy_Percent': [r['Accuracy_%'] for r in table_data]
    })
    print(acc_table.to_string(index=False))

    # Table 4: TPR at different FPR levels (ROC curve data from top-left)
    print(f"\n📈 ROC Performance Metrics")
    print("-" * 40)
    roc_table = pd.DataFrame({
        'Length_Category': [r['Length_Category'] for r in table_data],
        'AUC': [r['AUC'] for r in table_data],
        'TPR_at_FPR_0.01': [r['TPR_at_FPR_0.01'] for r in table_data],
        'TPR_at_FPR_0.1': [r['TPR_at_FPR_0.1'] for r in table_data]
    })
    print(roc_table.to_string(index=False, float_format='%.3f'))

    # Save to CSV if path provided
    if output_csv:
        df.to_csv(output_csv, index=False)

        # Also save the individual chart tables
        base_name = output_csv.replace('.csv', '')
        sample_table.to_csv(f"{base_name}_samples.csv", index=False)
        auc_table.to_csv(f"{base_name}_auc.csv", index=False)
        acc_table.to_csv(f"{base_name}_accuracy.csv", index=False)
        roc_table.to_csv(f"{base_name}_roc.csv", index=False)

        print(f"\nTables saved:")
        print(f"  Main: {output_csv}")
        print(f"  Samples: {base_name}_samples.csv")
        print(f"  AUC: {base_name}_auc.csv")
        print(f"  Accuracy: {base_name}_accuracy.csv")
        print(f"  ROC: {base_name}_roc.csv")

    return df


def main(args):
    print(f"=== LENGTH-BASED WATERMARK STRENGTH ANALYSIS ===")
    print(f"Target Language: {args.tgt_lang}")
    print(f"Base Directory: {args.base_wm_dir}")

    # First, analyze the length distribution and get percentiles
    print(f"\n=== TEXT LENGTH DISTRIBUTION ===")
    length_counts, length_examples, percentiles = analyze_length_distribution(args.base_wm_dir, args.tgt_lang)

    total_texts = sum(length_counts.values())
    for category in ["short", "medium", "long"]:
        count = length_counts[category]
        percentage = (count / total_texts * 100) if total_texts > 0 else 0
        examples = length_examples.get(category, [])
        example_str = f" (examples: {examples})" if examples else ""
        print(f"{category.capitalize()}: {count} texts ({percentage:.1f}%){example_str}")

    # Evaluate watermark strength by length using percentile-based classification
    print(f"\n=== WATERMARK STRENGTH BY TEXT LENGTH ===")
    results = evaluate_by_length_and_language(args.base_wm_dir, args.tgt_lang, percentiles)

    # Save and display results table
    if results:
        df = save_results_table(results, args.tgt_lang, args.output_csv)

        # Also print compact summary table
        print(f"\n=== COMPACT SUMMARY ===")
        print(f"{'Length':<8} {'Samples':<8} {'Accuracy':<10} {'AUC':<8} {'TPR@0.1':<8}")
        print("-" * 50)
        for category in ["short", "medium", "long"]:
            if category in results:
                r = results[category]
                print(f"{category:<8} {r['num_samples']:<8} {r['accuracy']:>7.1f}%  {r['auc']:>6.3f}  {r['tpr_01']:>6.3f}")

    # Generate visualization if requested
    if args.output_plot and results:
        plt.figure(figsize=(12, 8))

        # Plot 1: ROC curves by length
        plt.subplot(2, 2, 1)
        colors = ['red', 'blue', 'green']
        for i, (category, result) in enumerate(results.items()):
            if len(result['fpr']) > 0 and len(result['tpr']) > 0:
                plt.plot(result['fpr'], result['tpr'],
                        label=f"{category.capitalize()} (AUC = {result['auc']:.3f})",
                        color=colors[i], linewidth=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves by Text Length")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Accuracy by length
        plt.subplot(2, 2, 2)
        categories = list(results.keys())
        accuracies = [results[cat]['accuracy'] for cat in categories]
        plt.bar(categories, accuracies, color=['red', 'blue', 'green'])
        plt.ylabel("Language Detection Accuracy (%)")
        plt.title("Detection Accuracy by Text Length")
        plt.ylim(0, 100)

        # Plot 3: Sample distribution
        plt.subplot(2, 2, 3)
        samples = [results[cat]['num_samples'] for cat in categories]
        plt.bar(categories, samples, color=['red', 'blue', 'green'])
        plt.ylabel("Number of Samples")
        plt.title("Sample Distribution by Length")

        # Plot 4: AUC comparison
        plt.subplot(2, 2, 4)
        aucs = [results[cat]['auc'] for cat in categories]
        plt.bar(categories, aucs, color=['red', 'blue', 'green'])
        plt.ylabel("AUC Score")
        plt.title("AUC by Text Length")
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {args.output_plot}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze watermark strength by text length and language")
    parser.add_argument(
        "--base_wm_dir", type=str, required=True, help="Base directory for watermark files"
    )
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--output_plot", type=str, default=None, help="Output plot file (.png)")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV file for results table")

    args = parser.parse_args()
    main(args)