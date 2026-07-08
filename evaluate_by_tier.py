#!/usr/bin/env python3
"""
Evaluate detection results broken down by attack-language resource tier.

Given watermarked (positive) and human (negative) z-score files that carry an
`attack_tier` field (produced by the random-attack pipeline), this computes
AUC and TPR@1% overall and per tier (high / medium / low), for the STEAM
recovery result and — optionally — the no-defense baseline side by side.

Accepts multiple files per stream (e.g. one per source language) so results can
be pooled across all non-English source languages in a single call.

Reuses the metric helpers from eval_detection.py so numbers match the rest of
the pipeline.
"""

import json
import argparse
from collections import defaultdict

from sklearn.metrics import roc_auc_score, roc_curve

from eval_detection import tpr_at_fpr

TIER_ORDER = ["high", "medium", "low"]


def load_zscores(paths):
    """Load z-score records from one or more jsonl files -> list of dicts."""
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _z(rec):
    z = rec.get("z_score")
    return 0.0 if z is None else float(z)


def metrics(wm_recs, hm_recs):
    """AUC + TPR@10% + TPR@1% for a set of positive/negative records."""
    if len(wm_recs) == 0 or len(hm_recs) == 0:
        return None
    y_true = [0] * len(hm_recs) + [1] * len(wm_recs)
    y_scores = [_z(r) for r in hm_recs] + [_z(r) for r in wm_recs]
    # Need both classes present.
    if len(set(y_true)) < 2:
        return None
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    try:
        tpr01 = float(tpr_at_fpr(fpr, tpr, 0.01))
        tpr10 = float(tpr_at_fpr(fpr, tpr, 0.1))
    except Exception:
        tpr01 = tpr10 = float("nan")
    return {
        "n_wm": len(wm_recs),
        "n_hm": len(hm_recs),
        "auc": float(auc),
        "tpr@1%": tpr01,
        "tpr@10%": tpr10,
    }


def group_by_tier(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[r.get("attack_tier")].append(r)
    return groups


def evaluate_stream(wm_recs, hm_recs):
    """Return {tier -> metrics} plus an 'overall' entry."""
    out = {}
    wm_groups = group_by_tier(wm_recs)
    hm_groups = group_by_tier(hm_recs)
    for tier in TIER_ORDER:
        m = metrics(wm_groups.get(tier, []), hm_groups.get(tier, []))
        if m is not None:
            out[tier] = m
    out["overall"] = metrics(wm_recs, hm_recs)
    return out


def _fmt(v, nd=3):
    if v is None:
        return "  -  "
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_table(steam, baseline):
    rows = TIER_ORDER + ["overall"]
    have_base = baseline is not None
    header = f"{'tier':<9} {'n_wm':>6} {'n_hm':>6}"
    if have_base:
        header += f" | {'base_AUC':>9} {'STEAM_AUC':>10} {'ΔAUC':>7}"
        header += f" | {'base_T@1%':>10} {'STEAM_T@1%':>11} {'ΔT@1%':>8}"
    else:
        header += f" | {'AUC':>7} {'TPR@1%':>8} {'TPR@10%':>8}"
    print(header)
    print("-" * len(header))
    for tier in rows:
        s = steam.get(tier)
        if s is None:
            continue
        line = f"{tier:<9} {s['n_wm']:>6} {s['n_hm']:>6}"
        if have_base:
            b = baseline.get(tier)
            b_auc = b["auc"] if b else None
            b_t1 = b["tpr@1%"] if b else None
            d_auc = (s["auc"] - b_auc) if b_auc is not None else None
            d_t1 = (s["tpr@1%"] - b_t1) if b_t1 is not None else None
            line += f" | {_fmt(b_auc):>9} {_fmt(s['auc']):>10} {_fmt(d_auc):>7}"
            line += f" | {_fmt(b_t1):>10} {_fmt(s['tpr@1%']):>11} {_fmt(d_t1):>8}"
        else:
            line += f" | {_fmt(s['auc']):>7} {_fmt(s['tpr@1%']):>8} {_fmt(s['tpr@10%']):>8}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection by attack-language tier.")
    parser.add_argument("--steam_wm", nargs="+", required=True,
                        help="STEAM watermarked z-score file(s) (with attack_tier)")
    parser.add_argument("--steam_hm", nargs="+", required=True,
                        help="STEAM human z-score file(s)")
    parser.add_argument("--baseline_wm", nargs="+", default=None,
                        help="No-defense baseline watermarked z-score file(s)")
    parser.add_argument("--baseline_hm", nargs="+", default=None,
                        help="No-defense baseline human z-score file(s)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to dump the full results as JSON")
    args = parser.parse_args()

    steam = evaluate_stream(load_zscores(args.steam_wm), load_zscores(args.steam_hm))

    baseline = None
    if args.baseline_wm and args.baseline_hm:
        baseline = evaluate_stream(load_zscores(args.baseline_wm), load_zscores(args.baseline_hm))

    print("\n=== Detection by attack-language tier ===")
    print_table(steam, baseline)
    print()

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"steam": steam, "baseline": baseline}, f, indent=2)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
