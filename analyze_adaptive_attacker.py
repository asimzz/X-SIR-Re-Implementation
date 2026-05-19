#!/usr/bin/env python3
"""
Adaptive oracle-attacker analysis for the STEAM rebuttal.

For each watermarked text t and each pivot X in a 47-language attack pool,
we already have a STEAM-BO z-score z_X(t). The oracle attacker minimizes
per-text:

    adaptive_score(t) = min over X in pool of z_X(t)

The same min-aggregation is applied to the human side so the wm/hm score
distributions use the matched statistic, then we report AUC, TPR@FPR=1%,
TPR@FPR=10%, F1, and the calibration drift between the un-attacked
threshold (proxied by single-pivot fr) and the adaptive-min null.

Reads:
    gen/{model_abbr}/{method}_seed{seed}/mc4.{lang}.bo.z_score.jsonl
    gen/{model_abbr}/{method}_seed{seed}/mc4.{lang}.bo.hum.z_score.jsonl

Writes:
    {out_dir}/adaptive_attacker_summary.csv
    {out_dir}/adaptive_attacker_trace_{model}_{method}_seed{seed}.jsonl

Usage:
    python3 analyze_adaptive_attacker.py \\
        --model_abbr aya-23-8B --method kgw --seed 0 --num_texts 100
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_detection import tpr_at_fpr, f1_at_fpr


ATTACK_LANGS_47 = [
    # 17 original (NEW_SUPPORTED_LANGS)
    "fr", "de", "it", "es", "pt",
    "pl", "nl", "ru", "hi", "ko", "ja",
    "bn", "fa", "vi", "iw", "uk", "ta",
    # 30 new
    "ar", "da", "hr", "fi", "no", "sv", "hu", "cs", "el", "af",
    "bg", "ro", "sk", "sl", "lt", "lv", "et", "sr", "zh-CN", "sw",
    "zu", "yo", "tg", "ka", "ha", "ca", "be", "am", "tr", "id",
]


def read_z_scores(path, n=None):
    z_scores = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            obj = json.loads(line)
            z = obj.get("z_score")
            z_scores.append(0.0 if z is None else z)
    return z_scores


def read_prompts(path, n=None):
    prompts = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            obj = json.loads(line)
            prompts.append(obj.get("prompt", ""))
    return prompts


def load_pivot_matrix(run_dir, langs, side, num_texts=None):
    """Load Z matrix shape (n_texts, n_used_langs).

    side ∈ {"bo", "bo.hum"}. Skips langs whose file is missing (logs a
    warning) and truncates to the minimum row count if rows mismatch.
    """
    suffix = "bo.z_score.jsonl" if side == "bo" else "bo.hum.z_score.jsonl"
    columns = []
    used = []
    for lang in langs:
        path = os.path.join(run_dir, f"mc4.{lang}.{suffix}")
        if not os.path.isfile(path):
            print(f"[warn] missing {path}", file=sys.stderr)
            continue
        col = read_z_scores(path, n=num_texts)
        columns.append(col)
        used.append(lang)

    if not columns:
        raise FileNotFoundError(f"no BO files found in {run_dir} for side={side}")

    min_len = min(len(c) for c in columns)
    Z = np.array([c[:min_len] for c in columns], dtype=float).T  # (n_texts, n_langs)
    return Z, used


def compute_metrics(hm_scores, wm_scores, alpha=0.01, tau_fixed=None):
    """AUC + TPR/F1 at 1% and 10% FPR + 99th-pct human threshold.

    If `tau_fixed` is provided, also report TPR / FPR at that threshold —
    the operationally meaningful answer to "what does the detection rate
    look like at the un-attacked-calibrated threshold?"
    """
    n = min(len(hm_scores), len(wm_scores))
    hm_z = np.asarray(hm_scores[:n], dtype=float)
    wm_z = np.asarray(wm_scores[:n], dtype=float)

    y_true = [0] * n + [1] * n
    y_scores = list(hm_z) + list(wm_z)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    out = {
        "auc": float(auc),
        "tpr_at_1pct_fpr": float(tpr_at_fpr(fpr, tpr, alpha)),
        "tpr_at_10pct_fpr": float(tpr_at_fpr(fpr, tpr, 0.10)),
        "f1_at_1pct_fpr": float(f1_at_fpr(y_true, y_scores, alpha)),
        "f1_at_10pct_fpr": float(f1_at_fpr(y_true, y_scores, 0.10)),
        "hm_99th_pct": float(np.percentile(hm_z, 99)),
        "n_human": n,
        "n_watermarked": n,
    }

    if tau_fixed is not None:
        out["tpr_at_tau_fixed"] = float(np.mean(wm_z >= tau_fixed))
        out["fpr_at_tau_fixed"] = float(np.mean(hm_z >= tau_fixed))

    return out


def calibration_drift(hm_ref, hm_adaptive):
    """Drift diagnostic between un-attacked-proxy null and adaptive-min null."""
    hm_ref = np.asarray(hm_ref, dtype=float)
    hm_adaptive = np.asarray(hm_adaptive, dtype=float)
    tau_unattacked_proxy = float(np.percentile(hm_ref, 99))
    tau_adaptive_null = float(np.percentile(hm_adaptive, 99))
    drift = tau_adaptive_null - tau_unattacked_proxy
    observed_fpr = float(np.mean(hm_adaptive >= tau_unattacked_proxy))
    return {
        "tau_unattacked_proxy": tau_unattacked_proxy,
        "tau_adaptive_null": tau_adaptive_null,
        "calibration_drift": drift,
        "observed_fpr_at_tau_proxy": observed_fpr,
    }


def write_csv_summary(rows, out_path):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_text_trace(prompts, picked_langs, adapt_wm, Z_wm, used_langs, out_path, n=100):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = min(n, len(prompts), len(picked_langs), Z_wm.shape[0])
    with open(out_path, "w") as f:
        for i in range(n):
            row = {
                "row_idx": i,
                "prompt_prefix": (prompts[i] or "")[:120],
                "picked_lang": picked_langs[i],
                "adaptive_score": float(adapt_wm[i]),
                "z_per_lang": {lang: float(Z_wm[i, j]) for j, lang in enumerate(used_langs)},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_console_table(model, method, seed, n_pivots, n_texts, regime_metrics, drift):
    print()
    print(f"=== Adaptive attacker — {model} / {method} / seed {seed} (n_texts={n_texts}, n_pivots={n_pivots}) ===")
    print(f"τ_unattacked = {drift['tau_unattacked_proxy']:.3f}  "
          f"(99th pct of single-pivot reference human BO z; defender's calibrated threshold)")
    print()
    header = (f"{'regime':<22} {'AUC':>7} {'TPR@1%':>8} {'TPR@10%':>9} "
              f"{'TPR@τ_un':>9} {'FPR@τ_un':>9} {'F1@1%':>7} {'τ99(hm)':>9}")
    print(header)
    print("-" * len(header))
    for regime in ("single_pivot_ref", "single_pivot_mean", "single_pivot_worst", "adaptive_min"):
        m = regime_metrics.get(regime)
        if m is None:
            continue
        tpr_tau = m.get("tpr_at_tau_fixed", float("nan"))
        fpr_tau = m.get("fpr_at_tau_fixed", float("nan"))
        print(f"{regime:<22} {m['auc']:>7.3f} {m['tpr_at_1pct_fpr']:>8.3f} {m['tpr_at_10pct_fpr']:>9.3f} "
              f"{tpr_tau:>9.3f} {fpr_tau:>9.3f} {m['f1_at_1pct_fpr']:>7.3f} {m['hm_99th_pct']:>9.3f}")

    print()
    print("=== Calibration drift ===")
    print(f"τ_unattacked_proxy (99th pct of fr human bo z) : {drift['tau_unattacked_proxy']:.3f}")
    print(f"τ_adaptive_null    (99th pct of min-N hm bo z) : {drift['tau_adaptive_null']:.3f}")
    print(f"drift (τ_adaptive − τ_proxy)                   : {drift['calibration_drift']:+.3f}")
    print(f"observed FPR at τ_proxy under adaptive null    : {drift['observed_fpr_at_tau_proxy']:.3%}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Adaptive oracle-attacker rebuttal analysis")
    parser.add_argument("--gen_dir", default=os.path.join(here, "gen"))
    parser.add_argument("--model_abbr", default="aya-23-8B")
    parser.add_argument("--method", default="kgw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref_lang", default="fr",
                        help="Single-pivot reference for τ_unattacked proxy (default: fr)")
    parser.add_argument("--langs", nargs="+", default=ATTACK_LANGS_47,
                        help="Attack pivot pool (default: 47-pool = 17 original + 30 new)")
    parser.add_argument("--num_texts", type=int, default=100,
                        help="Subsample first N rows of every jsonl (default: 100)")
    parser.add_argument("--target_alpha", type=float, default=0.01)
    parser.add_argument("--trace_n", type=int, default=100)
    parser.add_argument("--out_dir", default=os.path.join(here, "results", "adaptive_attacker"))
    args = parser.parse_args()

    run_dir = os.path.join(args.gen_dir, args.model_abbr, f"{args.method}_seed{args.seed}")
    if not os.path.isdir(run_dir):
        print(f"run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] reading from {run_dir}")
    Z_wm, used = load_pivot_matrix(run_dir, args.langs, "bo", num_texts=args.num_texts)
    Z_hm, used_hm = load_pivot_matrix(run_dir, used, "bo.hum", num_texts=args.num_texts)

    if used != used_hm:
        # Re-align Z_wm columns to the lang set present on both sides.
        common = [l for l in used if l in set(used_hm)]
        wm_idx = [used.index(l) for l in common]
        hm_idx = [used_hm.index(l) for l in common]
        Z_wm = Z_wm[:, wm_idx]
        Z_hm = Z_hm[:, hm_idx]
        used = common

    n_texts = min(Z_wm.shape[0], Z_hm.shape[0])
    Z_wm = Z_wm[:n_texts, :]
    Z_hm = Z_hm[:n_texts, :]
    n_pivots = len(used)
    print(f"[info] Z_wm.shape={Z_wm.shape}, Z_hm.shape={Z_hm.shape}, n_pivots={n_pivots}")

    adapt_wm = Z_wm.min(axis=1)
    adapt_hm = Z_hm.min(axis=1)

    # Sanity: adaptive score must be ≤ every per-pivot score
    assert np.all(adapt_wm[:, None] <= Z_wm + 1e-9), "adapt_wm not ≤ Z_wm"
    assert np.all(adapt_hm[:, None] <= Z_hm + 1e-9), "adapt_hm not ≤ Z_hm"

    # Reference pivot for τ_unattacked (independent of attacker pool — always
    # loaded directly so that restricting --langs doesn't shift the threshold).
    ref_lang = args.ref_lang
    ref_path = os.path.join(run_dir, f"mc4.{ref_lang}.bo.hum.z_score.jsonl")
    if not os.path.isfile(ref_path):
        print(f"[warn] ref_lang {ref_lang} hm file missing; falling back to {used[0]}")
        ref_lang = used[0]
        ref_idx = used.index(ref_lang)
        ref_hm_z = Z_hm[:, ref_idx]
    else:
        ref_hm_z = np.asarray(read_z_scores(ref_path, n=args.num_texts), dtype=float)[:n_texts]

    tau_unattacked_proxy = float(np.percentile(ref_hm_z, 99))
    print(f"[info] τ_unattacked_proxy = 99th pct of {ref_lang} hm BO z = {tau_unattacked_proxy:.3f}")

    # Per-pivot metrics (with TPR at fixed τ_unattacked)
    per_pivot = {}
    for j, lang in enumerate(used):
        per_pivot[lang] = compute_metrics(Z_hm[:, j], Z_wm[:, j],
                                          alpha=args.target_alpha,
                                          tau_fixed=tau_unattacked_proxy)

    regimes = {}
    if ref_lang in per_pivot:
        regimes["single_pivot_ref"] = per_pivot[ref_lang]
    else:
        # Reference is outside attacker pool — compute its single-pivot regime separately
        wm_path = os.path.join(run_dir, f"mc4.{ref_lang}.bo.z_score.jsonl")
        if os.path.isfile(wm_path):
            wm_ref = np.asarray(read_z_scores(wm_path, n=args.num_texts), dtype=float)[:n_texts]
            regimes["single_pivot_ref"] = compute_metrics(ref_hm_z, wm_ref,
                                                          alpha=args.target_alpha,
                                                          tau_fixed=tau_unattacked_proxy)

    # Mean across the 47 single-pivot regimes
    means = {}
    for k in ("auc", "tpr_at_1pct_fpr", "tpr_at_10pct_fpr",
              "f1_at_1pct_fpr", "f1_at_10pct_fpr", "hm_99th_pct",
              "tpr_at_tau_fixed", "fpr_at_tau_fixed"):
        vals = [per_pivot[l][k] for l in used]
        means[k] = float(np.mean(vals))
    means["n_human"] = n_texts
    means["n_watermarked"] = n_texts
    regimes["single_pivot_mean"] = means

    # Worst single-pivot by AUC
    worst_lang = min(used, key=lambda l: per_pivot[l]["auc"])
    regimes["single_pivot_worst"] = {**per_pivot[worst_lang], "worst_lang": worst_lang}

    # Adaptive (matched min-min)
    regimes["adaptive_min"] = compute_metrics(adapt_hm, adapt_wm,
                                              alpha=args.target_alpha,
                                              tau_fixed=tau_unattacked_proxy)

    # Drift (uses ref_lang hm distribution, loaded independently of attacker pool)
    drift = calibration_drift(ref_hm_z, adapt_hm)

    # Picked-lang histogram
    picked_idx = Z_wm.argmin(axis=1)
    picked_langs = [used[j] for j in picked_idx]
    hist = Counter(picked_langs)

    # Verification §6: warn if one lang dominates
    top_lang, top_count = hist.most_common(1)[0]
    top_frac = top_count / n_texts
    if top_frac > 0.8:
        print(f"[warn] picked-lang concentration: {top_lang} dominates {top_frac:.0%} — "
              f"adaptive attacker is essentially single-best-pivot")

    # Verification §7: drift sign
    if drift["calibration_drift"] > 0:
        print(f"[warn] calibration_drift is positive ({drift['calibration_drift']:+.3f}) — "
              f"unexpected; expected ≤ 0 for min-aggregated null")

    # Write CSV
    rows = []
    for regime_name, m in regimes.items():
        row = {
            "model": args.model_abbr,
            "method": args.method,
            "seed": args.seed,
            "regime": regime_name,
            "n_pivots": n_pivots if regime_name == "adaptive_min" else 1,
            "n_texts": n_texts,
            "auc": m["auc"],
            "tpr_at_1pct_fpr": m["tpr_at_1pct_fpr"],
            "tpr_at_10pct_fpr": m["tpr_at_10pct_fpr"],
            "tpr_at_tau_unattacked": m.get("tpr_at_tau_fixed"),
            "fpr_at_tau_unattacked": m.get("fpr_at_tau_fixed"),
            "f1_at_1pct_fpr": m["f1_at_1pct_fpr"],
            "f1_at_10pct_fpr": m["f1_at_10pct_fpr"],
            "hm_99th_pct": m["hm_99th_pct"],
            "tau_unattacked_proxy": drift["tau_unattacked_proxy"],
            "tau_adaptive_null": drift["tau_adaptive_null"],
            "calibration_drift": drift["calibration_drift"],
            "observed_fpr_at_tau_proxy": drift["observed_fpr_at_tau_proxy"],
            "extra": m.get("worst_lang", ""),
        }
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "adaptive_attacker_summary.csv")
    write_csv_summary(rows, csv_path)
    print(f"[info] wrote {len(rows)} rows → {csv_path}")

    # Trace
    ref_path = os.path.join(run_dir, f"mc4.{ref_lang}.bo.z_score.jsonl")
    prompts = read_prompts(ref_path, n=args.num_texts)
    trace_path = os.path.join(
        args.out_dir,
        f"adaptive_attacker_trace_{args.model_abbr}_{args.method}_seed{args.seed}.jsonl",
    )
    write_per_text_trace(prompts, picked_langs, adapt_wm, Z_wm, used,
                         trace_path, n=args.trace_n)
    print(f"[info] wrote trace → {trace_path}")

    # Console
    print_console_table(args.model_abbr, args.method, args.seed,
                        n_pivots, n_texts, regimes, drift)

    # Picked-lang histogram (top 10)
    print()
    print("=== Picked-lang histogram (top 10) ===")
    for lang, count in hist.most_common(10):
        print(f"  {lang:<8} {count:>4}  ({count / n_texts:.1%})")


if __name__ == "__main__":
    main()
