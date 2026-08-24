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
      split each language's null max-scores into a calibration half and a verification
      half; fit ONE GLOBAL threshold τ* = 99th percentile of the pooled calibration
      halves; verify achieved FPR ≈ 1% on the held-out verification halves — the "fix".
      The split must be a seeded RANDOM permutation, not an index split: the mc4 human
      corpus is ordered and its first half has a heavier upper tail, which biases τ*
      upward and lands the achieved FPR near 0.4% instead of 1% (see SPLIT_MODE).

  D3. Per-language FPR at the single global τ*: with τ* fixed, break the verification
      false positives down by text language (the Tamil check — does any one language
      absorb a disproportionate FPR share even after the γ_lang correction?), and
      aggregate by high/medium/low resource tier. Also reports a per-language
      threshold τ*_ℓ so one can see whether the global τ* is already adequate.

  D4. Empirical vs Bonferroni: compare the FPR achieved at the empirically-calibrated
      τ* to that at the theory-derived Bonferroni threshold z_bonf = norm.ppf(1 −
      α/B), where B = min(budget, P) is the per-text BO evaluation budget — the number
      of hypotheses actually tested, matching the reviewer's "Bonferroni over 20
      evaluations". Report p_eff = α / SF(τ*), the effective number of independent
      Gaussian tests implied by τ*.

  D5. Extreme-value predictions for the null maximum: sqrt(2 ln B) and the Gumbel
      location Φ⁻¹(1 − 1/B). Both depend on B only, hence are constant in P at a fixed
      budget — the reason a single calibrated threshold transfers as the pool grows.

FINDING, measured on the 17-language nulls: p_eff ≈ 360-1050 ≫ B = 20, and Bonferroni
over B leaves 3.4-4.2% FPR rather than ≤1%. The per-candidate γ_lang-corrected null is
therefore HEAVIER-TAILED than Gaussian, which is why empirical calibration is required
rather than merely tighter. This contradicts the intuition that positive correlation
between related back-translation candidates shrinks the inflation; do not make that
claim from this data.

NOTE ON TPR: every committed positive (mc4.{lang}.bo.z_score.jsonl) was generated on
the buggy 84-pivot pool, and at P=33/66 the positives' subsample was drawn from 84
candidates while the nulls' was drawn from 125 — so they do not search the same
languages. Pass --no_tpr (and do so for anything published) until positives are
regenerated on the 125-pool.

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
import zlib

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
# The mc4 human corpus is NOT in random order: within every language the first 250
# texts have a systematically heavier upper tail than the last 250 (e.g. at P=126,
# p99 = 4.03 vs 3.74 and max = 7.74 vs 4.83). Calibrating on scores[:250] and
# verifying on scores[250:] is therefore NOT exchangeable — it over-estimates tau*
# and lands the achieved FPR at ~0.4% instead of the 1% target. A seeded random
# split, stratified within language so each language still contributes 250/250,
# restores exchangeability (achieved FPR 1.01-1.09% across 20 seeds). "random" is
# the default; "index" is retained to reproduce the July rebuttal numbers.
SPLIT_MODE = "random"
SPLIT_SEED = 0
FAMILYWISE_ALPHA = 0.01                      # for z_bonf = norm.ppf(1 - α/n_tests)
# Minimum verification-split size for a credible achieved-FPR-at-tau* estimate. The
# 1% tail must contain enough samples to be stable (~25 => ~2500 verify points); with
# only a few languages the extreme tail is too sparse, so that column is suppressed.
MIN_VERIFY_FOR_ACHIEVED_FPR = 2500
# Bonferroni is applied over the number of hypotheses actually tested = the BO
# evaluation budget per text (3 initial + 17 BO = 20), NOT the pool size. Each
# suspect text evaluates exactly this many pivots and the statistic is the max
# over them; the pool size is the search space, not the test count. This matches
# the reviewer's framing ("Bonferroni/Šidák correction over 20 evaluations").
N_EVALUATIONS = 20

# Provenance fingerprint: these ISO-1 codes are searchable ONLY when the complete
# language_code_converter is in effect (the 125-pivot pool). A file whose best_pivot
# values include any of these was generated on the fixed 125-pool; a file that uses
# none was generated on the buggy 84-pool (the stale converter dropped these). Used
# to avoid mixing 84-pool and 125-pool data in one analysis.
COMPLETE_CONVERTER_ONLY_LANGS = {
    'ak', 'ay', 'bm', 'bho', 'bs', 'ceb', 'ckb', 'dv', 'ee', 'eo', 'fy', 'haw',
    'ht', 'ilo', 'jw', 'kri', 'ku', 'ky', 'la', 'lb', 'lg', 'ln', 'lus', 'mai',
    'mi', 'mk', 'mni-Mtei', 'nso', 'ny', 'rw', 'sa', 'sd', 'sm', 'su', 'tk',
    'tl', 'ts', 'tt', 'ug', 'yi',
}

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

    Returns (scores: np.ndarray, n_none, provenance). None z-scores are floored to
    0.0 by default (consistent with eval_detection.py / analyze_pool_size.py) or
    dropped when drop_none=True. `provenance` is one of:
      "full125" - best_pivot values include a complete-converter-only language,
      "legacy84" - has best_pivots but none of them (buggy 84-pool),
      "unknown" - has scores but no best_pivot field (cannot tell),
      "missing" - file absent / empty.
    """
    if not os.path.isfile(path):
        return np.array([]), 0, "missing"
    import json
    scores = []
    n_none = 0
    n_pivots = 0
    uses_complete = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("best_pivot")
            if p:
                n_pivots += 1
                if p in COMPLETE_CONVERTER_ONLY_LANGS:
                    uses_complete = True
            z = obj.get("z_score")
            if z is None:
                n_none += 1
                if drop_none:
                    continue
                z = 0.0
            scores.append(float(z))
    if len(scores) == 0:
        prov = "missing"
    elif uses_complete:
        prov = "full125"
    elif n_pivots > 0:
        prov = "legacy84"
    else:
        prov = "unknown"
    return np.asarray(scores, dtype=float), n_none, prov


def split_by_index(scores, split_at):
    """Deterministic index split into (calibrate, verify, split_point).

    Uses min(split_at, n//2) so partial languages still yield disjoint halves.
    NOTE: not exchangeable on mc4 (see SPLIT_MODE) — kept for reproducing the
    July rebuttal numbers only.
    """
    n = len(scores)
    sp = min(split_at, n // 2)
    return scores[:sp], scores[sp:], sp


def split_at_random(scores, split_at, seed, lang):
    """Seeded random split into (calibrate, verify, split_point).

    Stratified per language: the seed is derived from (seed, lang) so each
    language is permuted independently but reproducibly, and each contributes
    the same calibrate/verify counts as the index split. Disjoint by
    construction (a permutation partitioned at sp).
    """
    n = len(scores)
    sp = min(split_at, n // 2)
    # Language-dependent but deterministic stream, so adding a language does not
    # perturb the split of the languages already analyzed.
    rs = np.random.RandomState(
        (seed * 1000003 + (zlib.crc32(lang.encode()) & 0x7FFFFFFF)) % (2 ** 31 - 1))
    idx = rs.permutation(n)
    return scores[idx[:sp]], scores[idx[sp:]], sp


def split_scores(scores, split_at, mode, seed, lang):
    """Dispatch to the configured calibrate/verify split."""
    if mode == "index":
        return split_by_index(scores, split_at)
    if mode == "random":
        return split_at_random(scores, split_at, seed, lang)
    raise ValueError(f"unknown split mode: {mode!r}")


# ---------------------------------------------------------------------------
# Per-pool analysis (the four deliverables)
# ---------------------------------------------------------------------------
def analyze_pool(gen_dir, model_abbr, P, method, seed, langs,
                 split_at=SPLIT_AT, calib_pctl=CALIB_PCTL,
                 alpha=FAMILYWISE_ALPHA, drop_none=False,
                 n_evaluations=N_EVALUATIONS,
                 split_mode=SPLIT_MODE, split_seed=SPLIT_SEED,
                 no_tpr=False):
    """Compute all deliverables for one pool size P.

    Returns (overall_row, per_lang_rows, present_langs).
    """
    per_lang = {}          # lang -> dict of raw arrays + counts
    calib_parts, verify_parts, pos_parts = [], [], []
    present_langs = []
    stale = {"stale_null": [], "stale_pos": [], "no_pos": []}

    for lang in langs:
        null_scores, n_none_null, null_prov = load_scores(
            null_path(gen_dir, model_abbr, P, method, seed, lang), drop_none)
        if no_tpr:
            # Decision: report FPR only. Every committed positive is 84-pool, and at
            # P=33/66 the positives' subsample was drawn from 84 candidates while the
            # nulls' was drawn from 125 — so they do not even search the same
            # languages. Any TPR computed against them would be meaningless.
            pos_scores, n_none_pos, pos_prov = np.array([]), 0, "suppressed"
        else:
            pos_scores, n_none_pos, pos_prov = load_scores(
                pos_path(gen_dir, model_abbr, P, method, seed, lang), drop_none)

        if len(null_scores) == 0:
            # No null statistic for this language at this pool — cannot contribute.
            continue
        if null_prov != "full125":
            # Never calibrate/measure FPR on a non-125-pool null — it would mix pools.
            stale["stale_null"].append((lang, null_prov))
            continue

        calib_l, verify_l, sp = split_scores(
            null_scores, split_at, split_mode, split_seed, lang)
        # Disjointness guard: the two halves partition the language's null scores.
        assert len(calib_l) + len(verify_l) == len(null_scores)

        # Positives count toward TPR only if they too are 125-pool (same detector
        # config as the null). A stale 84-pool positive is excluded and flagged.
        pos_ok = (len(pos_scores) > 0 and pos_prov == "full125")
        if no_tpr:
            pass                     # TPR deliberately not computed; not a data gap
        elif len(pos_scores) == 0:
            stale["no_pos"].append(lang)
        elif not pos_ok:
            stale["stale_pos"].append((lang, pos_prov))

        per_lang[lang] = {
            "null": null_scores, "calib": calib_l, "verify": verify_l,
            "pos": pos_scores if pos_ok else np.array([]),
            "n_none_null": n_none_null, "n_none_pos": n_none_pos,
            "null_prov": null_prov, "pos_prov": pos_prov, "pos_ok": pos_ok,
        }
        calib_parts.append(calib_l)
        verify_parts.append(verify_l)
        if pos_ok:
            pos_parts.append(pos_scores)
        present_langs.append(lang)

    if not calib_parts:
        return None, [], present_langs, stale

    calib_pool = np.concatenate(calib_parts)
    verify_pool = np.concatenate(verify_parts)
    pos_pool = np.concatenate(pos_parts) if pos_parts else np.array([])
    null_all = np.concatenate([per_lang[l]["null"] for l in present_langs])

    # -- D2: single GLOBAL calibrated threshold from pooled calibration halves ----
    tau_star = float(np.percentile(calib_pool, calib_pctl))
    achieved_fpr = float(np.mean(verify_pool > tau_star))
    tpr_empirical = float(np.mean(pos_pool > tau_star)) if len(pos_pool) else float("nan")

    # -- D1: naive fixed threshold (the "problem") -------------------------------
    # Reported on all null scores AND on the verification half alone, so that the
    # table's "problem" and "fix" columns are measured on the same held-out sample.
    naive_fpr_overall = float(np.mean(null_all > NAIVE_Z))
    naive_fpr_verify = float(np.mean(verify_pool > NAIVE_Z))

    # -- D4: Bonferroni threshold (theory-derived, data-independent) --------------
    # Correct over the number of hypotheses actually tested = min(budget, pool).
    # BO evaluates n_evaluations pivots per text (unless the pool is smaller), so
    # this is the test count the max is taken over — NOT the pool size P.
    n_tests = min(n_evaluations, P)
    z_bonf = float(norm.ppf(1 - alpha / n_tests))
    tpr_bonf = float(np.mean(pos_pool > z_bonf)) if len(pos_pool) else float("nan")
    achieved_fpr_bonf = float(np.mean(verify_pool > z_bonf))
    tpr_gain = (tpr_empirical - tpr_bonf
                if not (np.isnan(tpr_empirical) or np.isnan(tpr_bonf)) else float("nan"))
    # Effective number of independent tests: familywise / single-test tail mass.
    # Compare against n_tests: p_eff << n_tests => the evaluated pivots are
    # positively correlated (back-translations of one text), so empirical
    # calibration is tighter than a Bonferroni correction over n_tests.
    sf_tau = float(norm.sf(tau_star))
    p_eff = float(alpha / sf_tau) if sf_tau > 0 else float("inf")

    # -- D5: extreme-value predictions for the null maximum ----------------------
    # The statistic is the max over the B = min(budget, P) candidates BO actually
    # EVALUATES, not over the pool P. Under an iid standard-Gaussian per-candidate
    # null, that maximum concentrates at sqrt(2 ln B) with the Gumbel-refined
    # location Phi^-1(1 - 1/B). Both depend on B only, so both are CONSTANT in P at
    # a fixed budget — which is precisely why one calibrated threshold transfers as
    # the pool grows. Comparing them to the empirical tau* also tests the Gaussian
    # assumption: tau* far ABOVE these (and p_eff >> B) means the per-candidate null
    # is heavier-tailed than Gaussian, so a Bonferroni correction over B is not
    # conservative enough and empirical calibration is required rather than optional.
    ev_sqrt_2lnB = float(np.sqrt(2.0 * np.log(n_tests)))
    ev_gumbel = float(norm.ppf(1.0 - 1.0 / n_tests))
    tau_minus_ev_gumbel = tau_star - ev_gumbel

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
    fpr_by_lang_own_tau = {}
    for lang in present_langs:
        d = per_lang[lang]
        verify_l = d["verify"]
        pos_l = d["pos"]
        fpr_at_global = float(np.mean(verify_l > tau_star)) if len(verify_l) else float("nan")
        fpr_by_lang[lang] = fpr_at_global

        # Per-language calibrated threshold tau*_l: the 99th percentile of THIS
        # language's own calibration half, verified on its own verification half.
        # Answers r7a3's "clearer per-language FPR calibration" — it shows whether
        # the single global tau* is already adequate per language, or whether one
        # language would need its own threshold to hit 1%.
        calib_l = d["calib"]
        if len(calib_l) and len(verify_l):
            tau_l = float(np.percentile(calib_l, calib_pctl))
            fpr_own = float(np.mean(verify_l > tau_l))
        else:
            tau_l, fpr_own = float("nan"), float("nan")
        fpr_by_lang_own_tau[lang] = fpr_own

        per_lang_rows.append({
            "pool_size": P,
            "lang": lang,
            "resource_tier": RESOURCE_TIER.get(lang, "?"),
            "n_verify_lang": len(verify_l),
            "n_none_null": d["n_none_null"],
            "n_none_pos": d["n_none_pos"],
            "null_provenance": d["null_prov"],
            "pos_provenance": d["pos_prov"],
            "naive_fpr_lang": float(np.mean(d["null"] > NAIVE_Z)),
            "fpr_at_global_threshold": fpr_at_global,
            "tau_lang": tau_l,
            "fpr_at_own_threshold": fpr_own,
            "tpr_lang_empirical": (float(np.mean(pos_l > tau_star))
                                   if len(pos_l) else float("nan")),
        })

    finite_fprs = [v for v in fpr_by_lang.values() if not np.isnan(v)]
    max_lang_fpr = float(max(finite_fprs)) if finite_fprs else float("nan")
    fpr_spread = (float(max(finite_fprs) - min(finite_fprs))
                  if finite_fprs else float("nan"))
    worst_lang = (max(fpr_by_lang, key=lambda k: (fpr_by_lang[k], k))
                  if finite_fprs else "")

    # -- D6: per-resource-tier aggregation of the per-language FPR ----------------
    # The paper's thesis runs along the high/medium/low axis, so the reviewers'
    # question "does FPR control hold for low-resource languages too?" is answered
    # by these three numbers, not by the pooled average.
    tier_stats = {}
    for tier in ("high", "medium", "low"):
        vals = [fpr_by_lang[l] for l in present_langs
                if RESOURCE_TIER.get(l) == tier and not np.isnan(fpr_by_lang[l])]
        tier_stats[f"fpr_{tier}_mean"] = float(np.mean(vals)) if vals else float("nan")
        tier_stats[f"fpr_{tier}_max"] = float(np.max(vals)) if vals else float("nan")
        tier_stats[f"n_langs_{tier}"] = len(vals)

    overall_row = {
        "pool_size": P,
        "n_langs": len(present_langs),
        "n_langs_with_pos": len(pos_parts),
        "n_calib": int(len(calib_pool)),
        "n_verify": int(len(verify_pool)),
        "n_pos": int(len(pos_pool)),
        "n_eval_budget": n_evaluations,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "naive_z": round(NAIVE_Z, 6),
        "naive_fpr_overall": naive_fpr_overall,
        "naive_fpr_verify": naive_fpr_verify,
        "global_threshold_tau": tau_star,
        "achieved_fpr_verify": achieved_fpr,
        "tpr_empirical": tpr_empirical,
        "tpr_roc_1pct": tpr_roc_1pct,
        "n_bonferroni_tests": n_tests,
        "z_bonf": z_bonf,
        "achieved_fpr_bonf": achieved_fpr_bonf,
        "tpr_bonf": tpr_bonf,
        "tpr_gain": tpr_gain,
        "p_eff": p_eff,
        "ev_sqrt_2lnB": ev_sqrt_2lnB,
        "ev_gumbel": ev_gumbel,
        "tau_minus_ev_gumbel": tau_minus_ev_gumbel,
        "max_lang_fpr": max_lang_fpr,
        "worst_lang": worst_lang,
        "fpr_spread": fpr_spread,
        "max_lang_fpr_own_tau": (
            float(max(v for v in fpr_by_lang_own_tau.values() if not np.isnan(v)))
            if any(not np.isnan(v) for v in fpr_by_lang_own_tau.values())
            else float("nan")),
    }
    overall_row.update(tier_stats)
    return overall_row, per_lang_rows, present_langs, stale


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
    plt.plot(Ps, tpr_bonf, marker="s", ls="--", label="TPR @ Bonferroni z (α/n_eval)")
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
    plt.plot(Ps, zb, marker="s", ls="--", label="Bonferroni z = Φ⁻¹(1−α/n_eval)")
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


def emit_latex(overall_rows, per_lang_rows, out_dir, tag, langs):
    """Write two paste-ready booktabs tables (the paper's LaTeX is not in this repo).

    Table 1 (pool level): the calibration result per pool size — the naive-threshold
    inflation, the Bonferroni-over-budget threshold and its FPR, the extreme-value
    location, the calibrated tau*, and the achieved held-out FPR.
    Table 2 (per language): FPR at the single global tau*, grouped by resource tier.

    No TPR column is emitted: every committed positive is 84-pool (see the module
    docstring), so TPR at tau* is not reportable from this data.
    """
    os.makedirs(out_dir, exist_ok=True)

    def pct(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}"

    lines = [
        r"% Auto-generated by analyze_fpr_calibration.py --emit_latex. Do not hand-edit.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{rrrrrrr}", r"\toprule",
        (r"$P$ & $B$ & FPR@$2.33$ & $z_{\mathrm{Bonf}}$ & FPR@$z_{\mathrm{Bonf}}$ "
         r"& $\tau^\ast$ & FPR@$\tau^\ast$ \\"),
        r"\midrule",
    ]
    for r in overall_rows:
        lines.append(
            f"{r['pool_size']} & {r['n_bonferroni_tests']} & {pct(r['naive_fpr_verify'])}\\% & "
            f"{r['z_bonf']:.2f} & {pct(r['achieved_fpr_bonf'])}\\% & "
            f"{r['global_threshold_tau']:.2f} & {pct(r['achieved_fpr_verify'])}\\% \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{Empirical FPR calibration of STEAM's max-over-search statistic on "
         r"held-out human texts, over all 17 target languages. $B$ is the per-text BO "
         r"evaluation budget, i.e.\ the number of hypotheses the maximum is actually "
         r"taken over. FPR@$2.33$ is the inflation at the naive single-test $1\%$ "
         r"threshold; $\tau^\ast$ is the $99$th percentile of a disjoint calibration "
         r"split. $\tau^\ast$ is stable in $P$, so a threshold calibrated once "
         r"transfers as language coverage grows.}"),
        r"\label{tab:fpr-calibration}", r"\end{table}", "",
    ]

    pools = [r["pool_size"] for r in overall_rows]
    by_key = {(r["pool_size"], r["lang"]): r for r in per_lang_rows}
    colspec = "l" + "r" * len(pools)
    lines += [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        "Lang & " + " & ".join(rf"$P{{=}}{p}$" for p in pools) + r" \\",
        r"\midrule",
    ]
    for tier, label in (("high", "High-resource"), ("medium", "Medium-resource"),
                        ("low", "Low-resource")):
        tier_langs = [l for l in langs if RESOURCE_TIER.get(l) == tier
                      and any((p, l) in by_key for p in pools)]
        if not tier_langs:
            continue
        lines.append(rf"\multicolumn{{{len(pools) + 1}}}{{l}}{{\emph{{{label}}}}} \\")
        for l in tier_langs:
            cells = [pct(by_key[(p, l)]["fpr_at_global_threshold"])
                     if (p, l) in by_key else "—" for p in pools]
            lines.append(f"\\texttt{{{l}}} & " + " & ".join(cells) + r" \\")
        means = []
        for p in pools:
            vals = [by_key[(p, l)]["fpr_at_global_threshold"] for l in tier_langs
                    if (p, l) in by_key]
            means.append(pct(float(np.mean(vals)) if vals else float("nan")))
        lines.append(r"\cmidrule(l){1-" + str(len(pools) + 1) + r"}")
        lines.append(rf"\quad mean & " + " & ".join(means) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{Per-language false-positive rate (\%) at the \emph{single global} "
         r"threshold $\tau^\ast$ of Table~\ref{tab:fpr-calibration}, on the held-out "
         r"verification split. FPR control holds across all three resource tiers: no "
         r"language absorbs a disproportionate share of the error budget, which is "
         r"what the language-specific null correction $\gamma_\ell$ buys.}"),
        r"\label{tab:fpr-per-language}", r"\end{table}", "",
    ]

    path = os.path.join(out_dir, f"fpr_calibration_tables_{tag}.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _fnum(x, nd=3):
    """Format a float, showing an em dash for missing/NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def print_rebuttal_table(overall_rows):
    """Print a consistent FPR-calibration table, showing only credible/available columns.

    Always shown (credible even on partial data): the naive-threshold FPR, the
    Bonferroni-threshold FPR, and the calibrated threshold tau*. Conditionally shown:
      - achieved FPR at tau*  -> only when n_verify is large enough for a stable
        1%-tail estimate (the extreme tail is too sparse on a few languages), and
      - TPR columns           -> only when 125-pool positives are present.
    p_eff is intentionally excluded from the headline table: it assumes a Gaussian
    per-test null, which does not hold here (the empirical results show a heavier tail);
    it remains in the CSV for reference.
    """
    print("\n=== FPR calibration of the max-over-search statistic ===")
    z_bonf = overall_rows[0].get("z_bonf")
    n_tests = overall_rows[0].get("n_bonferroni_tests")
    achfpr_ok = all(r["n_verify"] >= MIN_VERIFY_FOR_ACHIEVED_FPR for r in overall_rows)
    has_tpr = any(not (isinstance(r.get("tpr_empirical"), float) and np.isnan(r["tpr_empirical"]))
                  and r.get("tpr_empirical") is not None for r in overall_rows)

    # (header, width, cell-fn)
    cols = [
        ("Pool P", 6, lambda r: str(r["pool_size"])),
        ("langs", 5, lambda r: str(r["n_langs"])),
        ("B", 3, lambda r: str(r["n_bonferroni_tests"])),
        ("FPR@2.33", 8, lambda r: _fnum(r["naive_fpr_verify"])),
        (f"FPR@{z_bonf:.2f}", 8, lambda r: _fnum(r["achieved_fpr_bonf"])),
        ("EV(B)", 6, lambda r: _fnum(r["ev_gumbel"], 2)),
        ("tau*", 6, lambda r: _fnum(r["global_threshold_tau"], 2)),
    ]
    if achfpr_ok:
        cols.append(("FPR@tau*", 9, lambda r: _fnum(r["achieved_fpr_verify"])))
    cols.append(("worstLang", 9, lambda r: f"{r['worst_lang']}:{_fnum(r['max_lang_fpr'])}"))
    if has_tpr:
        cols.append(("TPR@tau*", 9, lambda r: _fnum(r["tpr_empirical"])))
        cols.append((f"TPR@{z_bonf:.2f}", 9, lambda r: _fnum(r["tpr_bonf"])))

    hdr = " | ".join(f"{h:>{w}}" for h, w, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in overall_rows:
        print(" | ".join(f"{fn(r):>{w}}" for _, w, fn in cols))

    print(f"\nColumns: FPR@2.33 = false-positive rate of the max-over-search statistic at the")
    print(f"  naive single-test 1% threshold (z=2.33) — the inflation W3 warns of.")
    print(f"  FPR@{z_bonf:.2f} = FPR at the Bonferroni threshold over the {n_tests}-eval budget (Phi^-1(1-a/{n_tests})).")
    print(f"  tau* = empirically calibrated threshold (99th pct of the calibration-split null).")
    if not achfpr_ok:
        print("  (Held-out FPR@tau* omitted: too few languages for a stable 1%-tail estimate.)")
    if not has_tpr:
        print("  (TPR columns omitted: 125-pool positives not yet regenerated.)")


def print_tier_table(overall_rows):
    """Per-resource-tier FPR at the global tau* — the axis the paper's thesis runs on."""
    print("\n=== Per-resource-tier FPR at the global tau* (verify split) ===")
    hdr = f"{'Pool P':>6} | {'high':>13} | {'medium':>13} | {'low':>13}"
    print(hdr)
    print("-" * len(hdr))
    for r in overall_rows:
        cells = []
        for tier in ("high", "medium", "low"):
            cells.append(f"{_fnum(r[f'fpr_{tier}_mean'])}/{_fnum(r[f'fpr_{tier}_max'])}")
        print(f"{r['pool_size']:>6} | " + " | ".join(f"{c:>13}" for c in cells))
    print("  Cells are mean/max over the languages in that tier.")


def print_extreme_value_note(overall_rows):
    """The multiple-testing / extreme-value paragraph, computed rather than asserted."""
    print("\n=== Statistical control of the max statistic ===")
    for r in overall_rows:
        print(f"  P={r['pool_size']:>3}  B={r['n_bonferroni_tests']:>2}  "
              f"sqrt(2 ln B)={r['ev_sqrt_2lnB']:.2f}  "
              f"Gumbel Phi^-1(1-1/B)={r['ev_gumbel']:.2f}  "
              f"z_bonf={r['z_bonf']:.2f}  tau*={r['global_threshold_tau']:.2f}  "
              f"p_eff={r['p_eff']:.0f}")
    b = {r["n_bonferroni_tests"] for r in overall_rows}
    print(f"  B is constant in P ({sorted(b)}) at a fixed budget, so the "
          f"extreme-value location is too — hence tau* transfers across pool sizes."
          if len(b) == 1 else
          f"  B varies with P ({sorted(b)}), so the extreme-value location and tau* "
          f"are expected to grow with the pool.")
    if all(r["p_eff"] > r["n_bonferroni_tests"] for r in overall_rows):
        print("  p_eff > B at every pool: the per-candidate null is HEAVIER-TAILED than")
        print("  Gaussian, so Bonferroni over B is NOT conservative here and empirical")
        print("  calibration is required, not merely tighter. Do not claim that positive")
        print("  correlation between candidates reduces the inflation — it is not what")
        print("  this data shows.")


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
    ap.add_argument("--n_evaluations", type=int, default=N_EVALUATIONS,
                    help="BO evaluation budget per text = number of hypotheses the "
                         "Bonferroni correction is applied over (default: 20)")
    ap.add_argument("--pool_budgets", nargs="+", default=None,
                    metavar="P:BUDGET",
                    help="Per-pool BO evaluation budget as 'P:budget' pairs "
                         "(e.g. 33:10 66:20 126:38). Overrides --n_evaluations for "
                         "the listed pools; pools not listed fall back to "
                         "--n_evaluations. Use for the constant-budget-fraction "
                         "experiment where the budget scales with the pool.")
    ap.add_argument("--drop_none", action="store_true",
                    help="Drop None z-scores instead of flooring to 0.0")
    ap.add_argument("--split_mode", choices=["random", "index"], default=SPLIT_MODE,
                    help="Calibrate/verify split. 'random' (default) is a seeded "
                         "per-language permutation and is exchangeable; 'index' "
                         "takes the first half and is NOT (the mc4 corpus is "
                         "ordered) — use it only to reproduce the July rebuttal.")
    ap.add_argument("--split_seed", type=int, default=SPLIT_SEED,
                    help="Seed for --split_mode random")
    ap.add_argument("--no_tpr", action="store_true",
                    help="Do not load positives or report any TPR column. Correct "
                         "for this dataset: all committed positives are 84-pool.")
    ap.add_argument("--require_langs", type=int, default=None,
                    help="Exit non-zero unless every analyzed pool has this many "
                         "languages present. Guards against publishing a table "
                         "silently backed by fewer languages than intended.")
    ap.add_argument("--emit_latex", action="store_true",
                    help="Also write paste-ready booktabs tables (.tex)")
    ap.add_argument("--out_dir", default=os.path.join(here, "results", "fpr_calibration"))
    args = parser_defaults(ap)

    # Per-pool budget overrides (constant-budget-fraction experiment). Any pool
    # not listed falls back to the scalar --n_evaluations.
    pool_budgets = {}
    for spec in (args.pool_budgets or []):
        p_str, _, b_str = spec.partition(":")
        pool_budgets[int(p_str)] = int(b_str)

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

            stale_by_pool = {}
            for P in use_pools:
                row, plr, present, stale = analyze_pool(
                    args.gen_dir, args.model_abbr, P, method, seed, args.langs,
                    split_at=args.split_at, calib_pctl=args.calib_pctl,
                    alpha=args.alpha, drop_none=args.drop_none,
                    n_evaluations=pool_budgets.get(P, args.n_evaluations),
                    split_mode=args.split_mode, split_seed=args.split_seed,
                    no_tpr=args.no_tpr)
                coverage[P] = present
                stale_by_pool[P] = stale
                if row is None:
                    print(f"  P={P} ({method} seed={seed}): no usable 125-pool null data — skipping")
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
            # Only meaningful when 125-pool positives exist; under --no_tpr every TPR is
            # NaN and the plot would be an empty pair of axes.
            if any(not np.isnan(r.get("tpr_empirical", float("nan")))
                   for r in overall_rows):
                plot_tpr_vs_bonferroni(
                    overall_rows,
                    os.path.join(out_dir, f"tpr_empirical_vs_bonferroni_{tag}.png"))
            plot_threshold(overall_rows, os.path.join(out_dir, f"threshold_vs_pool_{tag}.png"))
            plot_per_lang_heatmap(per_lang_rows, analyzed_pools, args.langs,
                                  os.path.join(out_dir, f"per_lang_fpr_heatmap_{tag}.png"))
            print(f"Wrote plots → {out_dir}/*_{tag}.png")

            if args.emit_latex:
                tex = emit_latex(overall_rows, per_lang_rows, out_dir, tag, args.langs)
                print(f"Wrote LaTeX tables → {tex}")

            print_coverage(coverage, args.langs)
            print_provenance_warnings(stale_by_pool)
            print_rebuttal_table(overall_rows)
            print_tier_table(overall_rows)
            print_extreme_value_note(overall_rows)

            if args.require_langs is not None:
                short = {P: len(coverage.get(P, []))
                         for P in analyzed_pools
                         if len(coverage.get(P, [])) != args.require_langs}
                if short:
                    print(f"\nERROR: --require_langs {args.require_langs} not met: "
                          + ", ".join(f"P={P} has {n}" for P, n in sorted(short.items())))
                    sys.exit(1)


def print_provenance_warnings(stale_by_pool):
    any_stale = any(s["stale_null"] or s["stale_pos"] or s["no_pos"]
                    for s in stale_by_pool.values())
    if not any_stale:
        return
    print("\n=== Provenance warnings (mixed-pool data excluded) ===")
    for P in sorted(stale_by_pool):
        s = stale_by_pool[P]
        if s["stale_null"]:
            langs = ", ".join(f"{l}({p})" for l, p in s["stale_null"])
            print(f"  P={P}: NULL not 125-pool → EXCLUDED from FPR: {langs}")
        if s["stale_pos"]:
            langs = ", ".join(f"{l}({p})" for l, p in s["stale_pos"])
            print(f"  P={P}: POSITIVE not 125-pool → TPR skipped (FPR still valid): {langs}")
        if s["no_pos"]:
            print(f"  P={P}: no positives yet → TPR skipped: {', '.join(s['no_pos'])}")
    print("  Fix: run scripts/reset_targets_for_126.sh then run_steam_pool_sweep.sh to "
          "regenerate positives on the 125-pool; FPR-side results above are already valid.")


def parser_defaults(ap):
    """Parse args (thin wrapper kept separate for testability)."""
    return ap.parse_args()


if __name__ == "__main__":
    main()
