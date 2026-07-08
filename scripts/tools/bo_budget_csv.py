import argparse
import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from sklearn.metrics import roc_auc_score, roc_curve

from eval_detection import tpr_at_fpr
from utils import read_jsonl


LANGUAGES = [
    "fr", "de", "it", "es", "pt",
    "pl", "nl", "ru", "hi", "ko", "ja",
    "bn", "fa", "vi", "iw", "uk", "ta",
]

BUDGETS = [5, 10, 15, 20]


def bo_paths(base_dir, lang, budget):
    sub = "" if budget == 20 else f"budget_{budget}"
    folder = os.path.join(base_dir, sub) if sub else base_dir
    return (
        os.path.join(folder, f"mc4.{lang}.bo.z_score.jsonl"),
        os.path.join(folder, f"mc4.{lang}.bo.hum.z_score.jsonl"),
    )


def extract_zscores(items):
    return [x["z_score"] if x["z_score"] is not None else 0 for x in items]


def compute_metrics(wm_path, hum_path):
    wm = read_jsonl(wm_path)
    hum = read_jsonl(hum_path)
    n = min(len(wm), len(hum))
    if len(wm) != len(hum):
        print(
            f"warning: length mismatch wm={len(wm)} hum={len(hum)} "
            f"({wm_path}); truncating to {n}",
            file=sys.stderr,
        )
        wm, hum = wm[:n], hum[:n]
    y_true = [0] * len(hum) + [1] * len(wm)
    y_scores = extract_zscores(hum) + extract_zscores(wm)
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc, float(tpr_at_fpr(fpr, tpr, 0.01)), len(wm), len(hum)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate BO detector AUC and TPR@1% across budgets and languages into a CSV."
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.join(REPO_ROOT, "gen/aya-23-8B/kgw_seed0"),
        help="Root of the BO outputs (default: gen/aya-23-8B/kgw_seed0).",
    )
    parser.add_argument(
        "--out",
        default="bo_budget_ablation.csv",
        help="Output CSV path (default: bo_budget_ablation.csv).",
    )
    args = parser.parse_args()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "budget", "auc", "tpr_at_1pct", "n_mod", "n_hum"])
        for lang in LANGUAGES:
            for budget in BUDGETS:
                wm_path, hum_path = bo_paths(args.base_dir, lang, budget)
                if not (os.path.exists(wm_path) and os.path.exists(hum_path)):
                    print(
                        f"warning: missing files for {lang} budget={budget} "
                        f"(wm={os.path.exists(wm_path)}, hum={os.path.exists(hum_path)})",
                        file=sys.stderr,
                    )
                    writer.writerow([lang, budget, "", "", "", ""])
                    continue
                auc, tpr1, n_mod, n_hum = compute_metrics(wm_path, hum_path)
                writer.writerow([lang, budget, f"{auc:.6f}", f"{tpr1:.6f}", n_mod, n_hum])

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
