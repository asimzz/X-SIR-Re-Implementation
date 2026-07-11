#!/usr/bin/env python3
"""
FPR calibration of the STEAM-BO max-over-search statistic (rebuttal W3).

The concern (ovcy W3, r7a3, GoLu W2): STEAM takes the MAX z-score over a search of
candidate pivot languages, so a threshold that gives 1% FPR for a *single* detection
admits more than 1% FPR once you maximise over many candidates, and the inflation
grows with the pool size P. This script makes that inflation explicit and shows it is
controlled, producing four rebuttal deliverables from the TRUE null statistic:

  D1. Naive fixed-threshold FPR: apply the single-detection 1%-FPR threshold
      (z = norm.ppf(0.99) ≈ 2.326) to the null MAX statistic and report empirical
      FPR vs P — the "problem" (expected > 1% and rising with P).

  D2. Calibrated max-statistic threshold with a DISJOINT calibrate/verify split:
      split each language's null max-scores by index into a calibration half and a
      verification half; fit ONE GLOBAL threshold τ* = 99th percentile of the pooled
      calibration halves; verify achieved FPR ≈ 1% on the held-out verification halves
      and report TPR retained on positives at τ* — the "fix".

  D3. Per-language FPR at the single global τ*: with τ* fixed, break the verification
      false positives down by text language (the Tamil check — does any one language
      absorb a disproportionate FPR share even after the γ_lang correction?).

  D4. Empirical vs Bonferroni: compare TPR at the empirically-calibrated τ* to TPR at
      the theory-derived Bonferroni threshold z_bonf = norm.ppf(1 − α/P). Report the
      effective number of independent tests p_eff = α / SF(τ*); p_eff ≪ P quantifies
      that back-translation z-scores are positively correlated, so empirical
      calibration is tighter than the conservative Bonferroni/Šidák alternative.

IMPORTANT: this reads the INDEPENDENT-search null files
  mc4.{lang}.bo.hum.indep.z_score.jsonl
(produced by steam_bo_detector.py --independent_human), NOT the paired-pivot
mc4.{lang}.bo.hum.z_score.jsonl files, which borrow the watermarked text's pivot and
therefore understate the null. Positives (mc4.{lang}.bo.z_score.jsonl) are the
max-over-own-search statistic and are used as-is.

Disjointness note: γ_lang is fit from the mc4.{lang}.val.jsonl corpus, which is
disjoint from the mc4 human test texts that produce these null max-scores. So both the
calibration and verification halves are disjoint from the γ_lang calibration set by
construction, in addition to being disjoint from each other (index split).

Usage:
    python3 analyze_fpr_calibration.py --model_abbr aya-23-8B --methods kgw --seeds 0
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pool_size import read_z_scores, DEFAULT_LANGS  # loader + 17-lang list
from eval_detection import tpr_at_fpr                        # ROC cross-check only

# ---------------------------------------------------------------------------
# Fixed, reproducible configuration
# ---------------------------------------------------------------------------
NAIVE_ALPHA = 0.01
NAIVE_Z = float(norm.ppf(1 - NAIVE_ALPHA))   # 2.32634… — the single-test 1%-FPR point
CALIB_PCTL = 99.0                            # 99th percentile => 1% target FPR
SPLIT_AT = 250                               # first SPLIT_AT = calibrate, rest = verify
FAMILYWISE_ALPHA = 0.01                      # for z_bonf = norm.ppf(1 - α/P)

# Resource tier of each of the 17 target languages (for the per-lang table).
RESOURCE_TIER = {
    "fr": "high", "de": "high", "it": "high", "es": "high", "pt": "high",
    "pl": "medium", "nl": "medium", "ru": "medium", "hi": "medium",
    "ko": "medium", "ja": "medium",
    "bn": "low", "fa": "low", "vi": "low", "iw": "low", "uk": "low", "ta": "low",
}


# ---------------------------------------------------------------------------
# Data-loading layer
# ---------------------------------------------------------------------------
def discover_pools(gen_dir, model_abbr):
    """Return sorted list of integer pool sizes P from gen/{model}/pool_* dirs."""
    pattern = os.path.join(gen_dir, model_abbr, "pool_*")
    pools = []
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        if base.startswith("pool_"):
            try:
                pools.append(int(base[len("pool_"):]))
            except ValueError:
                continue
    return sorted(set(pools))


def null_path(gen_dir, model_abbr, P, method, seed, lang):
    return os.path.join(
        gen_dir, model_abbr, f"pool_{P}", f"{method}_seed{seed}",
        f"mc4.{lang}.bo.hum.indep.z_score.jsonl",
    )


def pos_path(gen_dir, model_abbr, P, method, seed, lang):
    return os.path.join(
        gen_dir, model_abbr, f"pool_{P}", f"{method}_seed{seed}",
        f"mc4.{lang}.bo.z_score.jsonl",
    )


def load_scores(path, drop_none=False):
    """Load z-scores from a jsonl file.

    Returns (scores: np.ndarray, n_total, n_none). None z-scores are floored to
    0.0 by default (consistent with eval_detection.py / analyze_pool_size.py) or
    dropped when drop_none=True. read_z_scores already maps None -> 0.0, so we
    count None via a raw pass when drop_none is requested.
    """
    if not os.path.isfile(path):
        return np.array([]), 0, 0
    import json
    scores = []
    n_none = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            z = json.loads(line).get("z_score")
            if z is None:
                n_none += 1
                if drop_none:
                    continue
                z = 0.0
            scores.append(float(z))
    return np.asarray(scores, dtype=float), len(scores) + (0 if not drop_none else n_none), n_none


def split_by_index(scores, split_at):
    """Deterministic index split into (calibrate, verify, split_point).

    Uses min(split_at, n//2) so partial languages still yield disjoint halves.
    """
    n = len(scores)
    sp = min(split_at, n // 2)
    return scores[:sp], scores[sp:], sp


# ---------------------------------------------------------------------------
# Per-pool analysis (the four deliverables)
# ---------------------------------------------------------------------------
def analyze_pool(gen_dir, model_abbr, P, method, seed, langs,
                 split_at=SPLIT_AT, calib_pctl=CALIB_PCTL,
                 alpha=FAMILYWISE_ALPHA, drop_none=False):
    """Compute all deliverables for one pool size P.

    Returns (overall_row, per_lang_rows, present_langs).
    """
    per_lang = {}          # lang -> dict of raw arrays + counts
    calib_parts, verify_parts, pos_parts = [], [], []
    present_langs = []

    for lang in langs:
        null_scores, _, n_none_null = load_scores(
            null_path(gen_dir, model_abbr, P, method, seed, lang), drop_none)
        pos_scores, _, n_none_pos = load_scores(
            pos_path(gen_dir, model_abbr, P, method, seed, lang), drop_none)

        if len(null_scores) == 0:
            # No null statistic for this language at this pool — cannot contribute.
            continue

        calib_l, verify_l, sp = split_by_index(null_scores, split_at)
        # Disjointness guard (calib and verify index ranges never overlap).
        assert len(calib_l) + len(verify_l) == len(null_scores)

        per_lang[lang] = {
            "null": null_scores, "calib": calib_l, "verify": verify_l,
            "pos": pos_scores, "n_none_null": n_none_null, "n_none_pos": n_none_pos,
        }
        calib_parts.append(calib_l)
        verify_parts.append(verify_l)
        if len(pos_scores) > 0:
            pos_parts.append(pos_scores)
        present_langs.append(lang)

    if not calib_parts:
        return None, [], present_langs

    calib_pool = np.concatenate(calib_parts)
    verify_pool = np.concatenate(verify_parts)
    pos_pool = np.concatenate(pos_parts) if pos_parts else np.array([])
    null_all = np.concatenate([per_lang[l]["null"] for l in present_langs])

    # -- D2: single GLOBAL calibrated threshold from pooled calibration halves ----
    tau_star = float(np.percentile(calib_pool, calib_pctl))
    achieved_fpr = float(np.mean(verify_pool > tau_star))
    tpr_empirical = float(np.mean(pos_pool > tau_star)) if len(pos_pool) else float("nan")

    # -- D1: naive fixed threshold (the "problem") -------------------------------
    naive_fpr_overall = float(np.mean(null_all > NAIVE_Z))

    # -- D4: Bonferroni threshold (theory-derived, data-independent) --------------
    z_bonf = float(norm.ppf(1 - alpha / P))
    tpr_bonf = float(np.mean(pos_pool > z_bonf)) if len(pos_pool) else float("nan")
    achieved_fpr_bonf = float(np.mean(verify_pool > z_bonf))
    tpr_gain = (tpr_empirical - tpr_bonf
                if not (np.isnan(tpr_empirical) or np.isnan(tpr_bonf)) else float("nan"))
    # Effective number of independent tests: familywise / single-test tail mass.
    sf_tau = float(norm.sf(tau_star))
    p_eff = float(alpha / sf_tau) if sf_tau > 0 else float("inf")

    # -- ROC cross-check: TPR@1% FPR on (verify null vs positives) ---------------
    tpr_roc_1pct = float("nan")
    if len(pos_pool) and len(verify_pool):
        from sklearn.metrics import roc_curve
        y_true = [0] * len(verify_pool) + [1] * len(pos_pool)
        y_scores = list(verify_pool) + list(pos_pool)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        try:
            tpr_roc_1pct = float(tpr_at_fpr(fpr, tpr, 0.01))
        except Exception:
            pass

    # -- D3: per-language FPR at the ONE global threshold + per-lang rows ---------
    per_lang_rows = []
    fpr_by_lang = {}
    for lang in present_langs:
        d = per_lang[lang]
        verify_l = d["verify"]
        pos_l = d["pos"]
        fpr_at_global = float(np.mean(verify_l > tau_star)) if len(verify_l) else float("nan")
        fpr_by_lang[lang] = fpr_at_global
        per_lang_rows.append({
            "pool_size": P,
            "lang": lang,
            "resource_tier": RESOURCE_TIER.get(lang, "?"),
            "n_verify_lang": len(verify_l),
            "n_none_null": d["n_none_null"],
            "n_none_pos": d["n_none_pos"],
            "naive_fpr_lang": float(np.mean(d["null"] > NAIVE_Z)),
            "fpr_at_global_threshold": fpr_at_global,
            "tpr_lang_empirical": (float(np.mean(pos_l > tau_star))
                                   if len(pos_l) else float("nan")),
        })

    finite_fprs = [v for v in fpr_by_lang.values() if not np.isnan(v)]
    max_lang_fpr = float(max(finite_fprs)) if finite_fprs else float("nan")
    fpr_spread = (float(max(finite_fprs) - min(finite_fprs))
                  if finite_fprs else float("nan"))

    overall_row = {
        "pool_size": P,
        "n_langs": len(present_langs),
        "n_calib": int(len(calib_pool)),
        "n_verify": int(len(verify_pool)),
        "n_pos": int(len(pos_pool)),
        "naive_z": round(NAIVE_Z, 6),
        "naive_fpr_overall": naive_fpr_overall,
        "global_threshold_tau": tau_star,
        "achieved_fpr_verify": achieved_fpr,
        "tpr_empirical": tpr_empirical,
        "tpr_roc_1pct": tpr_roc_1pct,
        "z_bonf": z_bonf,
        "achieved_fpr_bonf": achieved_fpr_bonf,
        "tpr_bonf": tpr_bonf,
        "tpr_gain": tpr_gain,
        "p_eff": p_eff,
        "max_lang_fpr": max_lang_fpr,
        "fpr_spread": fpr_spread,
    }
    return overall_row, per_lang_rows, present_langs


# ---------------------------------------------------------------------------
# Output: CSVs, plots, printed table
# ---------------------------------------------------------------------------
def write_csv(rows, out_path, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_naive_fpr(overall_rows, out_path):
    Ps = [r["pool_size"] for r in overall_rows]
    naive = [r["naive_fpr_overall"] for r in overall_rows]
    achieved = [r["achieved_fpr_verify"] for r in overall_rows]
    plt.figure(figsize=(5, 4))
    plt.plot(Ps, naive, marker="o", label="Naive threshold (z=2.326)")
    plt.plot(Ps, achieved, marker="s", label="Calibrated τ* (verify)")
    plt.axhline(0.01, color="gray", ls="--", lw=1, label="target FPR = 1%")
    plt.xlabel("Candidate pool size P")
    plt.ylabel("Empirical FPR of the max statistic")
    plt.title("FPR inflation vs pool size (problem → fix)")
    plt.xticks(Ps)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_tpr_vs_bonferroni(overall_rows, out_path):
    Ps = [r["pool_size"] for r in overall_rows]
    tpr_emp = [r["tpr_empirical"] for r in overall_rows]
    tpr_bonf = [r["tpr_bonf"] for r in overall_rows]
    plt.figure(figsize=(5, 4))
    plt.plot(Ps, tpr_emp, marker="o", label="TPR @ empirical τ* (true 1% FPR)")
    plt.plot(Ps, tpr_bonf, marker="s", ls="--", label="TPR @ Bonferroni z (α/P)")
    plt.xlabel("Candidate pool size P")
    plt.ylabel("TPR")
    plt.title("Empirical calibration vs Bonferroni")
    plt.xticks(Ps)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_threshold(overall_rows, out_path):
    Ps = [r["pool_size"] for r in overall_rows]
    tau = [r["global_threshold_tau"] for r in overall_rows]
    zb = [r["z_bonf"] for r in overall_rows]
    plt.figure(figsize=(5, 4))
    plt.plot(Ps, tau, marker="o", label="Empirical τ* (global 99th pct)")
    plt.plot(Ps, zb, marker="s", ls="--", label="Bonferroni z = Φ⁻¹(1−α/P)")
    plt.axhline(NAIVE_Z, color="gray", ls=":", lw=1, label=f"naive z = {NAIVE_Z:.3f}")
    plt.xlabel("Candidate pool size P")
    plt.ylabel("Decision threshold (z)")
    plt.title("Threshold vs pool size")
    plt.xticks(Ps)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_lang_heatmap(per_lang_rows, pools, langs, out_path):
    lang_order = [l for l in langs if any(r["lang"] == l for r in per_lang_rows)]
    if not lang_order or not pools:
        return
    mat = np.full((len(lang_order), len(pools)), np.nan)
    idx = {l: i for i, l in enumerate(lang_order)}
    pcol = {p: j for j, p in enumerate(pools)}
    for r in per_lang_rows:
        if r["lang"] in idx and r["pool_size"] in pcol:
            mat[idx[r["lang"]], pcol[r["pool_size"]]] = r["fpr_at_global_threshold"]
    plt.figure(figsize=(1.6 + 1.1 * len(pools), 0.42 * len(lang_order) + 1.5))
    im = plt.imshow(mat, aspect="auto", cmap="Reds", vmin=0.0,
                    vmax=max(0.02, np.nanmax(mat)))
    plt.colorbar(im, label="FPR at global τ*")
    plt.xticks(range(len(pools)), [f"P={p}" for p in pools])
    plt.yticks(range(len(lang_order)), lang_order)
    for i in range(len(lang_order)):
        for j in range(len(pools)):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]*100:.1f}", ha="center", va="center",
                         fontsize=7,
                         color="white" if mat[i, j] > 0.015 else "black")
    plt.title("Per-language FPR at the global 1% threshold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_rebuttal_table(overall_rows):
    print("\n=== FPR calibration of the max-over-search statistic ===")
    hdr = (f"{'Pool P':>7} | {'NaiveFPR@2.326':>14} | {'tau*':>7} | "
           f"{'AchFPR':>7} | {'TPR(emp)':>8} | {'z_bonf':>7} | "
           f"{'TPR(bonf)':>9} | {'dTPR':>7} | {'p_eff':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in overall_rows:
        print(f"{r['pool_size']:>7} | {r['naive_fpr_overall']:>14.3f} | "
              f"{r['global_threshold_tau']:>7.3f} | {r['achieved_fpr_verify']:>7.3f} | "
              f"{r['tpr_empirical']:>8.3f} | {r['z_bonf']:>7.3f} | "
              f"{r['tpr_bonf']:>9.3f} | {r['tpr_gain']:>+7.3f} | {r['p_eff']:>7.1f}")
    print("\nReading: NaiveFPR@2.326 > 1% and rising with P = the inflation W3 warns of.")
    print("         AchFPR ≈ 1% at the empirically-calibrated τ* = the fix holds on held-out nulls.")
    print("         TPR(emp) > TPR(bonf) and p_eff ≪ P = candidate z-scores are correlated,")
    print("         so empirical calibration is tighter than the conservative Bonferroni bound.")


def print_coverage(coverage, langs):
    print("\n=== Coverage (present langs / requested per pool) ===")
    for P in sorted(coverage):
        present = coverage[P]
        missing = [l for l in langs if l not in present]
        msg = f"  P={P}: {len(present)}/{len(langs)} langs"
        if missing:
            msg += f"   MISSING: {', '.join(missing)}"
        print(msg)


# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="FPR calibration of the STEAM-BO max statistic")
    ap.add_argument("--gen_dir", default=os.path.join(here, "gen"))
    ap.add_argument("--model_abbr", default="aya-23-8B")
    ap.add_argument("--methods", nargs="+", default=["kgw"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--pool_sizes", nargs="+", type=int, default=None,
                    help="Pools to analyze (default: auto-discover gen/{model}/pool_*)")
    ap.add_argument("--split_at", type=int, default=SPLIT_AT,
                    help="Index split: first N null scores calibrate, rest verify")
    ap.add_argument("--calib_pctl", type=float, default=CALIB_PCTL)
    ap.add_argument("--alpha", type=float, default=FAMILYWISE_ALPHA)
    ap.add_argument("--drop_none", action="store_true",
                    help="Drop None z-scores instead of flooring to 0.0")
    ap.add_argument("--out_dir", default=os.path.join(here, "results", "fpr_calibration"))
    args = parser_defaults(ap)

    pools = args.pool_sizes or discover_pools(args.gen_dir, args.model_abbr)
    if not pools:
        print(f"No pool_* directories under {args.gen_dir}/{args.model_abbr}. "
              f"Run scripts/run_steam_fpr_null.sh first.")
        return

    # Analyze each (method, seed) separately; default single method+seed.
    for method in args.methods:
        for seed in args.seeds:
            overall_rows, per_lang_rows, coverage = [], [], {}

            # Deduplicate pools whose data dir is the same physical path. The
            # canonical full pool (pool_126) and the legacy alias (pool_133) both
            # symlink to the base run, so without this they would produce two
            # identical full-pool rows. Sorted order keeps the smaller label (126).
            seen_real, use_pools = {}, []
            for P in pools:
                real = os.path.realpath(
                    os.path.join(args.gen_dir, args.model_abbr,
                                 f"pool_{P}", f"{method}_seed{seed}"))
                if real in seen_real:
                    print(f"  P={P} resolves to the same data as P={seen_real[real]} "
                          f"— skipping duplicate full-pool alias")
                    continue
                seen_real[real] = P
                use_pools.append(P)

            for P in use_pools:
                row, plr, present = analyze_pool(
                    args.gen_dir, args.model_abbr, P, method, seed, args.langs,
                    split_at=args.split_at, calib_pctl=args.calib_pctl,
                    alpha=args.alpha, drop_none=args.drop_none)
                coverage[P] = present
                if row is None:
                    print(f"  P={P} ({method} seed={seed}): no null data found — skipping")
                    continue
                overall_rows.append(row)
                per_lang_rows.extend(plr)

            if not overall_rows:
                print(f"No null data for {method} seed={seed}.")
                continue

            tag = f"{args.model_abbr}_{method}_seed{seed}"
            out_dir = args.out_dir
            overall_csv = os.path.join(out_dir, f"fpr_calibration_overall_{tag}.csv")
            per_lang_csv = os.path.join(out_dir, f"fpr_calibration_per_lang_{tag}.csv")
            write_csv(overall_rows, overall_csv)
            write_csv(per_lang_rows, per_lang_csv)
            print(f"\nWrote {len(overall_rows)} pool rows → {overall_csv}")
            print(f"Wrote {len(per_lang_rows)} per-lang rows → {per_lang_csv}")

            analyzed_pools = [r["pool_size"] for r in overall_rows]
            plot_naive_fpr(overall_rows, os.path.join(out_dir, f"naive_fpr_vs_pool_{tag}.png"))
            plot_tpr_vs_bonferroni(overall_rows, os.path.join(out_dir, f"tpr_empirical_vs_bonferroni_{tag}.png"))
            plot_threshold(overall_rows, os.path.join(out_dir, f"threshold_vs_pool_{tag}.png"))
            plot_per_lang_heatmap(per_lang_rows, analyzed_pools, args.langs,
                                  os.path.join(out_dir, f"per_lang_fpr_heatmap_{tag}.png"))
            print(f"Wrote plots → {out_dir}/*_{tag}.png")

            print_coverage(coverage, args.langs)
            print_rebuttal_table(overall_rows)


def parser_defaults(ap):
    """Parse args (thin wrapper kept separate for testability)."""
    return ap.parse_args()


if __name__ == "__main__":
    main()
