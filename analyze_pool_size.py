#!/usr/bin/env python3
"""
Aggregate STEAM-BO pool-size sweep outputs and produce FPR-rebuttal artifacts.

Walks   gen/{model_abbr}/pool_{N}/{method}_seed{seed}/
        mc4.{lang}.bo.z_score.jsonl
        mc4.{lang}.bo.hum.z_score.jsonl
for N ∈ {33, 66, 133}, computes TPR@FPR and the empirical 99th-percentile
human-side threshold (the null-side drift diagnostic), and writes a CSV plus
two PNG plots.

Usage:
    python3 analyze_pool_size.py --model_abbr llama-3.2-1B
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_detection import tpr_at_fpr


POOL_SIZES = [33, 66, 133]

DEFAULT_LANGS = [
    "fr", "de", "it", "es", "pt",
    "pl", "nl", "ru", "hi", "ko", "ja",
    "bn", "fa", "vi", "iw", "uk", "ta",
]


def read_z_scores(path):
    z_scores = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            z = obj.get("z_score")
            z_scores.append(0.0 if z is None else z)
    return z_scores


def compute_metrics(hm_path, wm_path):
    hm_z = read_z_scores(hm_path)
    wm_z = read_z_scores(wm_path)

    if len(hm_z) == 0 or len(wm_z) == 0:
        return None
    if len(hm_z) != len(wm_z):
        n = min(len(hm_z), len(wm_z))
        hm_z = hm_z[:n]
        wm_z = wm_z[:n]

    y_true = [0] * len(hm_z) + [1] * len(wm_z)
    y_scores = hm_z + wm_z

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    return {
        "tpr_at_1pct_fpr": float(tpr_at_fpr(fpr, tpr, 0.01)),
        "tpr_at_10pct_fpr": float(tpr_at_fpr(fpr, tpr, 0.10)),
        "threshold_99th_pct": float(np.percentile(hm_z, 99)),
        "auc": float(auc),
        "n_human": len(hm_z),
        "n_watermarked": len(wm_z),
    }


def collect_runs(gen_dir, model_abbr, methods, seeds, langs):
    rows = []
    for pool_size in POOL_SIZES:
        for method in methods:
            for seed in seeds:
                run_dir = os.path.join(
                    gen_dir, model_abbr,
                    f"pool_{pool_size}",
                    f"{method}_seed{seed}",
                )
                if not os.path.isdir(run_dir):
                    continue
                for lang in langs:
                    wm_path = os.path.join(run_dir, f"mc4.{lang}.bo.z_score.jsonl")
                    hm_path = os.path.join(run_dir, f"mc4.{lang}.bo.hum.z_score.jsonl")
                    if not (os.path.isfile(wm_path) and os.path.isfile(hm_path)):
                        continue

                    metrics = compute_metrics(hm_path, wm_path)
                    if metrics is None:
                        continue

                    rows.append({
                        "pool_size": pool_size,
                        "method": method,
                        "seed": seed,
                        "lang": lang,
                        **metrics,
                    })
    return rows


def write_csv(rows, out_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, value_key):
    """Group by (method, pool_size); return {method: {pool_size: (mean, std, n)}}."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["pool_size"])].append(r[value_key])

    out = defaultdict(dict)
    for (method, pool_size), vals in groups.items():
        arr = np.array(vals, dtype=float)
        out[method][pool_size] = (
            arr.mean(),
            arr.std(ddof=1) if len(arr) > 1 else 0.0,
            len(arr),
        )
    return out


def plot_tpr(agg, out_path):
    plt.figure(figsize=(5, 4))
    for method, by_pool in sorted(agg.items()):
        xs = sorted(by_pool.keys())
        means = [by_pool[x][0] for x in xs]
        stds = [by_pool[x][1] for x in xs]
        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=method)
    plt.xlabel("Candidate pool size N")
    plt.ylabel("TPR @ 1% FPR")
    plt.title("TPR @ 1% FPR vs candidate pool size")
    plt.xticks(POOL_SIZES)
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_threshold(agg, out_path):
    plt.figure(figsize=(5, 4))
    for method, by_pool in sorted(agg.items()):
        xs = sorted(by_pool.keys())
        means = [by_pool[x][0] for x in xs]
        stds = [by_pool[x][1] for x in xs]
        plt.errorbar(xs, means, yerr=stds, marker="s", capsize=3, label=method)
    plt.xlabel("Candidate pool size N")
    plt.ylabel("Empirical 99th-pct human z-score")
    plt.title("Null-side threshold drift vs pool size")
    plt.xticks(POOL_SIZES)
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_summary(agg_tpr, agg_thresh):
    print("\n=== TPR @ 1% FPR (mean ± std across seeds × langs) ===")
    print(f"{'method':<8} " + " ".join(f"pool_{p:<5}" for p in POOL_SIZES))
    for method in sorted(agg_tpr):
        cells = []
        for p in POOL_SIZES:
            if p in agg_tpr[method]:
                m, s, n = agg_tpr[method][p]
                cells.append(f"{m:.3f}±{s:.3f} (n={n})")
            else:
                cells.append("—")
        print(f"{method:<8} " + " ".join(f"{c:<14}" for c in cells))

    print("\n=== Empirical 99th-pct human z-score (drift diagnostic) ===")
    print(f"{'method':<8} " + " ".join(f"pool_{p:<5}" for p in POOL_SIZES))
    for method in sorted(agg_thresh):
        cells = []
        for p in POOL_SIZES:
            if p in agg_thresh[method]:
                m, s, n = agg_thresh[method][p]
                cells.append(f"{m:.3f}±{s:.3f}")
            else:
                cells.append("—")
        print(f"{method:<8} " + " ".join(f"{c:<14}" for c in cells))


DEFAULT_METHODS = ["kgw"]
DEFAULT_SEEDS = [0]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Pool-size FPR sweep analysis")
    parser.add_argument("--gen_dir", default=os.path.join(here, "gen"),
                        help="Path to gen/ directory (default: ./gen)")
    parser.add_argument("--model_abbr", default="llama-3.2-1B")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS,
                        help="Target languages to aggregate over")
    parser.add_argument("--out_dir", default=os.path.join(here, "results"),
                        help="Output directory (default: ./results)")
    args = parser.parse_args()

    rows = collect_runs(
        args.gen_dir, args.model_abbr,
        args.methods, args.seeds, args.langs,
    )

    if not rows:
        print("No matching pool-sweep outputs found. "
              "Run scripts/run_steam_pool_sweep.sh first.")
        return

    csv_path = os.path.join(args.out_dir, "pool_sweep.csv")
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows → {csv_path}")

    agg_tpr = aggregate(rows, "tpr_at_1pct_fpr")
    agg_thresh = aggregate(rows, "threshold_99th_pct")

    plot_tpr(agg_tpr, os.path.join(args.out_dir, "pool_sweep_tpr.png"))
    plot_threshold(agg_thresh, os.path.join(args.out_dir, "pool_sweep_threshold.png"))
    print(f"Wrote plots → {args.out_dir}/pool_sweep_{{tpr,threshold}}.png")

    print_summary(agg_tpr, agg_thresh)


if __name__ == "__main__":
    main()
