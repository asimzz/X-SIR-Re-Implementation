#!/usr/bin/env python3
"""Forecast the 17-language proportional-budget FPR calibration table.

Table 3 of ``results/fpr_calibration_writeup.md`` (constant 30.3% budget fraction)
is partial: at P=33 (B=10) and P=126 (B=38) only the 5 high-resource languages have
independent-search nulls; the other 12 are still generating. P=66 (B=20) is the
17-language fixed-budget run, reused.

This script predicts what the completed 17-language numbers will be, with an
uncertainty band, so the camera-ready text can be drafted before the sweep lands.

Estimator (per pending pool P):

  1. For the 5 high-resource languages we hold BOTH the fixed-budget nulls (B=20)
     and the proportional-budget nulls (B=10 or 38) over the SAME 500 texts. Fit a
     monotone quantile-to-quantile map T_P: fixed score -> proportional score on
     their pooled scores, with linear tail extrapolation so the top of the
     distribution (which drives tau*) is not clipped.
  2. Apply T_P to each of the 12 missing languages' fixed-budget scores at pool P
     to synthesise their proportional-budget scores.
  3. Pool 5 observed + 12 synthesised languages and compute exactly the statistics
     analyze_fpr_calibration.analyze_pool computes.

The band comes from a hierarchical bootstrap over (a) which languages support the
transfer map, (b) texts within a language, and (c) a per-language transfer residual
estimated leave-one-out across the 5 high-resource languages.

A flat additive tier offset (17-lang minus high-5 metric, measured on the
fixed-budget data at the same pool and split settings) is reported as an
independent cross-check, and the band is widened to cover it.

At P=66 the two budgets coincide, so T_66 is the identity and the whole pipeline
must reproduce the measured 17-language values; that runs as a self-check.

Once the 12 missing languages land, re-running fills the `actual` column and scores
the forecast.
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import norm

from analyze_fpr_calibration import (
    CALIB_PCTL,
    FAMILYWISE_ALPHA,
    NAIVE_Z,
    RESOURCE_TIER,
    SPLIT_AT,
    SPLIT_MODE,
    SPLIT_SEED,
    load_scores,
    null_path,
    split_scores,
    write_csv,
)
from analyze_pool_size import DEFAULT_LANGS

# The proportional-budget design: budget = 30.3% (== 20/66) of the pool.
POOL_BUDGETS = {33: 10, 66: 20, 126: 38}
FIXED_BUDGET = 20
HIGH_LANGS = [l for l in DEFAULT_LANGS if RESOURCE_TIER.get(l) == "high"]

# `naive_fpr_pct` is the verify-half figure the writeup's tables report;
# `naive_fpr_overall_pct` is the same quantity over all 500 texts per language. On
# a 5-language subset the verify half is only 1250 points (~1.3pp of binomial
# noise), so the overall version is the better-behaved estimand and the more
# trustworthy tier-offset cross-check — both are reported.
METRICS = ("naive_fpr_pct", "naive_fpr_overall_pct", "tau_star",
           "fpr_at_tau_pct", "bonf_fpr_pct")

# Per-language columns. `fpr_at_global_tau_pct` depends on the pooled tau*, so it
# is a forecast for EVERY language once any language is synthesised -- including
# the five that are already measured.
LANG_METRICS = ("naive_fpr_lang_pct", "naive_fpr_lang_overall_pct",
                "fpr_at_global_tau_pct", "tau_lang")


# ---------------------------------------------------------------------------
# Statistics — mirrors analyze_fpr_calibration.analyze_pool, but on in-memory
# score arrays so it can be applied to synthesised languages too.
# ---------------------------------------------------------------------------
_SPLIT_IDX_CACHE = {}


def _split_indices(lang, n, split_at, split_mode, split_seed):
    """Cached calibrate/verify index sets for one language.

    `split_scores` re-seeds a RandomState and permutes on every call; the
    bootstrap calls it a few hundred thousand times with the same (lang, n), so
    the permutation is memoised. Derived from `split_scores` itself so the two can
    never drift apart.
    """
    key = (lang, n, split_at, split_mode, split_seed)
    if key not in _SPLIT_IDX_CACHE:
        marker = np.arange(n, dtype=float)
        calib_idx, verify_idx, _ = split_scores(marker, split_at, split_mode,
                                                split_seed, lang)
        _SPLIT_IDX_CACHE[key] = (calib_idx.astype(int), verify_idx.astype(int))
    return _SPLIT_IDX_CACHE[key]


def pool_stats(lang_scores, P, n_evaluations,
               split_at=SPLIT_AT, calib_pctl=CALIB_PCTL,
               alpha=FAMILYWISE_ALPHA, split_mode=SPLIT_MODE, split_seed=SPLIT_SEED):
    """Compute the calibration statistics for one pool from {lang: scores}."""
    calib_parts, verify_parts, all_parts = [], [], []
    per_lang = {}
    for lang in sorted(lang_scores):
        scores = np.asarray(lang_scores[lang], dtype=float)
        if len(scores) == 0:
            continue
        # Bootstrap keys are "<lang>#<i>"; the split must follow the real language.
        calib_idx, verify_idx = _split_indices(
            lang.split("#", 1)[0], len(scores), split_at, split_mode, split_seed)
        calib_parts.append(scores[calib_idx])
        verify_parts.append(scores[verify_idx])
        all_parts.append(scores)
        per_lang[lang] = (scores[calib_idx], scores[verify_idx], scores)
    if not calib_parts:
        return None

    calib_pool = np.concatenate(calib_parts)
    verify_pool = np.concatenate(verify_parts)
    null_all = np.concatenate(all_parts)

    tau_star = float(np.percentile(calib_pool, calib_pctl))
    n_tests = min(n_evaluations, P)
    z_bonf = float(norm.ppf(1 - alpha / n_tests))
    sf_tau = float(norm.sf(tau_star))

    # Per-language view, mirroring analyze_fpr_calibration's per-lang CSV: the
    # language's own naive FPR, its FPR at the ONE global tau*, and the tau* it
    # would get on its own calibration half (reported as a robustness contrast --
    # the writeup shows per-language thresholds are worse than the pooled one).
    lang_stats = {}
    for lang, (calib_l, verify_l, all_l) in per_lang.items():
        tau_l = float(np.percentile(calib_l, calib_pctl))
        lang_stats[lang] = {
            "naive_fpr_lang_pct": 100.0 * float(np.mean(verify_l > NAIVE_Z)),
            "naive_fpr_lang_overall_pct": 100.0 * float(np.mean(all_l > NAIVE_Z)),
            "fpr_at_global_tau_pct": 100.0 * float(np.mean(verify_l > tau_star)),
            "tau_lang": tau_l,
            "n_verify_lang": len(verify_l),
        }

    return {
        "per_lang": lang_stats,
        "n_langs": len(calib_parts),
        "n_calib": len(calib_pool),
        "n_verify": len(verify_pool),
        "n_tests": n_tests,
        "naive_fpr_pct": 100.0 * float(np.mean(verify_pool > NAIVE_Z)),
        "naive_fpr_overall_pct": 100.0 * float(np.mean(null_all > NAIVE_Z)),
        "tau_star": tau_star,
        "fpr_at_tau_pct": 100.0 * float(np.mean(verify_pool > tau_star)),
        "bonf_fpr_pct": 100.0 * float(np.mean(verify_pool > z_bonf)),
        "z_bonf": z_bonf,
        "p_eff": float(alpha / sf_tau) if sf_tau > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Quantile transfer map
# ---------------------------------------------------------------------------
class QuantileTransfer:
    """Monotone map from fixed-budget scores to proportional-budget scores.

    Fitted as the order-statistic correspondence between two equal-length pooled
    samples, with linear extrapolation above the fitted range driven by the slope
    of the top `tail_frac` of order statistics — the calibration threshold sits at
    the 99th percentile, so a clipped upper tail would bias tau* downward.
    """

    def __init__(self, src, dst, tail_frac=0.01):
        src = np.sort(np.asarray(src, dtype=float))
        dst = np.sort(np.asarray(dst, dtype=float))
        # Put both samples on a common quantile grid so unequal sizes still work.
        n = min(len(src), len(dst))
        probs = (np.arange(n) + 0.5) / n
        self.src = np.quantile(src, probs)
        self.dst = np.quantile(dst, probs)
        k = max(2, int(round(tail_frac * n)))
        self.hi_slope = self._slope(self.src[-k:], self.dst[-k:])
        self.lo_slope = self._slope(self.src[:k], self.dst[:k])

    @staticmethod
    def _slope(x, y):
        dx = x[-1] - x[0]
        if dx <= 0:
            return 1.0
        return float((y[-1] - y[0]) / dx)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.interp(x, self.src, self.dst)
        above = x > self.src[-1]
        below = x < self.src[0]
        out[above] = self.dst[-1] + self.hi_slope * (x[above] - self.src[-1])
        out[below] = self.dst[0] + self.lo_slope * (x[below] - self.src[0])
        return out


def pooled(lang_scores, langs):
    parts = [lang_scores[l] for l in langs if len(lang_scores.get(l, ()))]
    return np.concatenate(parts) if parts else np.array([])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tree(gen_dir, model_abbr, P, method, seed, langs, require_full125=True):
    """Return {lang: scores} for every language whose null file is present & clean."""
    out = {}
    for lang in langs:
        scores, _, prov = load_scores(null_path(gen_dir, model_abbr, P, method, seed, lang))
        if len(scores) == 0:
            continue
        if require_full125 and prov != "full125":
            print(f"  ! skipping {model_abbr} P={P} {lang}: provenance {prov}", file=sys.stderr)
            continue
        out[lang] = scores
    return out


# ---------------------------------------------------------------------------
# Forecast for one pool
# ---------------------------------------------------------------------------
def loo_residuals(fixed_high, prop_high):
    """Per-language additive transfer residuals, leave-one-out over high-5.

    For each high-resource language, fit the transfer map on the other four,
    synthesise that language's proportional scores from its fixed ones, and take
    the median gap to the actual proportional scores (order-statistic aligned).
    The spread of these residuals is how much a single language deviates from the
    common transfer — the main source of error when extrapolating to 12 unseen
    languages.
    """
    langs = sorted(set(fixed_high) & set(prop_high))
    residuals = []
    for held in langs:
        others = [l for l in langs if l != held]
        if not others:
            continue
        tmap = QuantileTransfer(pooled(fixed_high, others), pooled(prop_high, others))
        synth = np.sort(tmap(fixed_high[held]))
        actual = np.sort(prop_high[held])
        n = min(len(synth), len(actual))
        probs = (np.arange(n) + 0.5) / n
        residuals.append(float(np.median(np.quantile(actual, probs)
                                         - np.quantile(synth, probs))))
    return np.asarray(residuals, dtype=float)


def loo_metric_errors(fixed_high, prop_high, P, budget, tau_ref):
    """Measured prediction error of the synthesis, per language, leave-one-out.

    The bootstrap over map/text/language-shift does not capture per-text search
    noise: the transfer map reproduces a language's score *distribution*, but the
    correlation between a text's fixed-budget and proportional-budget score is only
    ~0.7, so which particular texts land in the tail differs. Rather than model that
    with an additive noise term (which would over-disperse the marginal the map was
    fitted to match), measure the end-to-end error directly: hold out one
    high-resource language, fit the map on the other four, synthesise the held-out
    language, and compare its per-language metrics to the truth. The spread of these
    errors is the honest accuracy of this procedure for one unseen language.
    """
    langs = sorted(set(fixed_high) & set(prop_high))
    errs = {m: [] for m in LANG_METRICS}
    for held in langs:
        others = [l for l in langs if l != held]
        if not others:
            continue
        tmap = QuantileTransfer(pooled(fixed_high, others), pooled(prop_high, others))
        pred = pool_stats({held: tmap(fixed_high[held])}, P, budget)["per_lang"][held]
        truth = pool_stats({held: prop_high[held]}, P, budget)["per_lang"][held]
        # fpr_at_global_tau is defined against the POOLED tau*, not the language's
        # own, so recompute both sides at the reference tau*.
        for name, scores in (("pred", tmap(fixed_high[held])), ("truth", prop_high[held])):
            _, verify_idx = _split_indices(held, len(scores), SPLIT_AT,
                                           SPLIT_MODE, SPLIT_SEED)
            val = 100.0 * float(np.mean(scores[verify_idx] > tau_ref))
            (pred if name == "pred" else truth)["fpr_at_global_tau_pct"] = val
        for m in LANG_METRICS:
            errs[m].append(truth[m] - pred[m])
    return {m: np.asarray(v, dtype=float) for m, v in errs.items()}


def forecast_pool(P, fixed_all, prop_high, missing_langs, n_boot, rng):
    """Point forecast + bootstrap band for the 17-language proportional stats."""
    budget = POOL_BUDGETS[P]
    high = sorted(set(fixed_all) & set(prop_high))
    fixed_high = {l: fixed_all[l] for l in high}

    tmap = QuantileTransfer(pooled(fixed_high, high), pooled(prop_high, high))
    synth = {l: tmap(fixed_all[l]) for l in missing_langs}
    point = pool_stats({**prop_high, **synth}, P, budget)

    resid = loo_residuals(fixed_high, prop_high)

    samples = {m: [] for m in METRICS}
    lang_samples = {l: {m: [] for m in LANG_METRICS}
                    for l in list(prop_high) + list(missing_langs)}
    for _ in range(n_boot):
        # (a) which languages support the transfer map, and (b) texts within them.
        # Fixed and proportional nulls for a language cover the SAME 500 texts, so
        # resample text indices jointly to preserve that pairing.
        boot_fixed, boot_prop = {}, {}
        for i, l in enumerate(rng.choice(high, size=len(high), replace=True)):
            n = min(len(fixed_high[l]), len(prop_high[l]))
            idx = rng.integers(0, n, size=n)
            key = f"{l}#{i}"
            boot_fixed[key] = fixed_high[l][idx]
            boot_prop[key] = prop_high[l][idx]
        btmap = QuantileTransfer(pooled(boot_fixed, list(boot_fixed)),
                                 pooled(boot_prop, list(boot_prop)))

        # (c) per-language transfer residual for each synthesised language.
        boot_synth = {}
        for l in missing_langs:
            shift = rng.choice(resid) if len(resid) else 0.0
            boot_synth[l] = btmap(fixed_all[l]) + shift

        # The 5 observed languages enter the pool as measured, resampled by text.
        boot_obs = {}
        for l in high:
            idx = rng.integers(0, len(prop_high[l]), size=len(prop_high[l]))
            boot_obs[l] = prop_high[l][idx]

        st = pool_stats({**boot_obs, **boot_synth}, P, budget)
        for m in METRICS:
            samples[m].append(st[m])
        for l, ls in st["per_lang"].items():
            for m in LANG_METRICS:
                lang_samples[l][m].append(ls[m])

    def pctl(vals, metric=None):
        lo = float(np.percentile(vals, 2.5))
        hi = float(np.percentile(vals, 97.5))
        if metric and metric.endswith("_pct"):
            lo, hi = max(lo, 0.0), min(hi, 100.0)   # a rate cannot leave [0, 100]
        return lo, hi

    band = {m: pctl(samples[m], m) for m in METRICS}

    # Widen every SYNTHESISED language's band by the measured leave-one-out error
    # of the synthesis itself (the five observed languages need no such widening --
    # their scores are measured).
    loo_err = loo_metric_errors(fixed_high, prop_high, P, budget, point["tau_star"])
    lang_band = {}
    for l, v in lang_samples.items():
        entry = {}
        for m in LANG_METRICS:
            if not v[m]:
                continue
            vals = np.asarray(v[m], dtype=float)
            if l in missing_langs and len(loo_err[m]):
                vals = (vals[:, None] + loo_err[m][None, :]).ravel()
            entry[m] = pctl(vals, m)
        lang_band[l] = entry
    return point, band, lang_band, tmap


def tier_offset(fixed_all, P, high, missing_langs):
    """Cross-check: 17-lang minus high-5 metric on the fixed-budget data at pool P."""
    st_all = pool_stats(fixed_all, P, FIXED_BUDGET)
    st_high = pool_stats({l: fixed_all[l] for l in high}, P, FIXED_BUDGET)
    return {m: st_all[m] - st_high[m] for m in METRICS}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def lang_row(budget_name, P, B, lang, status, st_lang, band=None, actual_lang=None):
    row = dict(budget=budget_name, P=P, B=B, lang=lang,
               resource_tier=RESOURCE_TIER.get(lang, ""), status=status,
               n_verify_lang=st_lang["n_verify_lang"])
    for m in LANG_METRICS:
        lo, hi = (band or {}).get(m, (st_lang[m], st_lang[m]))
        row[m] = fmt(st_lang[m])
        row[f"{m}_low"] = fmt(lo)
        row[f"{m}_high"] = fmt(hi)
        actual = (actual_lang or {}).get(m)
        row[f"{m}_actual"] = fmt(actual)
    return row


def build_rows(gen_dir, args, rng):
    langs = list(DEFAULT_LANGS)
    rows = []
    lang_rows = []
    md_pred = {}

    # ---- fixed budget: complete, measured -------------------------------------
    fixed_by_pool = {}
    for P in args.pool_sizes:
        fixed_all = load_tree(gen_dir, args.model_abbr, P, args.method, args.seed, langs)
        fixed_by_pool[P] = fixed_all
        st = pool_stats(fixed_all, P, FIXED_BUDGET)
        for lang in langs:
            if lang in st["per_lang"]:
                lang_rows.append(lang_row("fixed", P, FIXED_BUDGET, lang, "measured",
                                          st["per_lang"][lang],
                                          actual_lang=st["per_lang"][lang]))
        for m in METRICS:
            rows.append(dict(
                budget="fixed", P=P, B=FIXED_BUDGET, status="final",
                n_langs_observed=st["n_langs"], n_langs_synthesised=0,
                metric=m, observed_5lang="", expected=fmt(st[m]),
                low=fmt(st[m]), high=fmt(st[m]), estimator="measured",
                actual=fmt(st[m]), in_band="",
                notes=f"{st['n_langs']} langs, B={FIXED_BUDGET}",
            ))

    # ---- proportional budget ---------------------------------------------------
    for P in args.pool_sizes:
        budget = POOL_BUDGETS[P]
        prop_all = load_tree(gen_dir, args.budgetfrac_abbr, P, args.method, args.seed, langs)
        fixed_all = fixed_by_pool[P]
        high = sorted(set(fixed_all) & set(prop_all) & set(HIGH_LANGS))
        missing = [l for l in fixed_all if l not in prop_all]

        st_high_prop = pool_stats({l: prop_all[l] for l in high}, P, budget) if high else None

        if not missing:
            # Complete — report the measurement.
            st = pool_stats(prop_all, P, budget)
            for lang in langs:
                if lang in st["per_lang"]:
                    lang_rows.append(lang_row("proportional", P, budget, lang,
                                              "measured", st["per_lang"][lang],
                                              actual_lang=st["per_lang"][lang]))
            for m in METRICS:
                rows.append(dict(
                    budget="proportional", P=P, B=budget, status="final",
                    n_langs_observed=st["n_langs"], n_langs_synthesised=0,
                    metric=m,
                    observed_5lang=fmt(st_high_prop[m]) if st_high_prop else "",
                    expected=fmt(st[m]), low=fmt(st[m]), high=fmt(st[m]),
                    estimator="measured", actual=fmt(st[m]), in_band="",
                    notes=f"{st['n_langs']} langs, B={budget}",
                ))
            md_pred[P] = dict(status="final", B=budget, stats=st, band=None)
            continue

        if not high:
            print(f"  ! P={P}: no high-resource proportional nulls — cannot forecast",
                  file=sys.stderr)
            continue

        point, band, lang_band, _ = forecast_pool(
            P, fixed_all, {l: prop_all[l] for l in high}, missing, args.n_boot, rng)

        for lang in langs:
            if lang not in point["per_lang"]:
                continue
            measured = lang in high
            lb = dict(lang_band.get(lang, {}))
            if measured:
                # This language's own scores are measured -- only the columns that
                # depend on the pooled tau* (which 12 unseen languages will move)
                # remain uncertain.
                for m in ("naive_fpr_lang_pct", "naive_fpr_lang_overall_pct", "tau_lang"):
                    lb[m] = (point["per_lang"][lang][m], point["per_lang"][lang][m])
            lang_rows.append(lang_row(
                "proportional", P, budget, lang,
                "measured(global-tau forecast)" if measured else "synthesised",
                point["per_lang"][lang], lb))
        offset = tier_offset(fixed_all, P, high, missing)
        cross = {m: st_high_prop[m] + offset[m] for m in METRICS}

        for m in METRICS:
            lo, hi = band[m]
            lo, hi = min(lo, cross[m]), max(hi, cross[m])
            rows.append(dict(
                budget="proportional", P=P, B=budget, status="forecast",
                n_langs_observed=len(high), n_langs_synthesised=len(missing),
                metric=m, observed_5lang=fmt(st_high_prop[m]),
                expected=fmt(point[m]), low=fmt(lo), high=fmt(hi),
                estimator="qq_transfer(band widened to tier_offset)",
                actual="", in_band="",
                notes=(f"tier_offset cross-check {fmt(cross[m])} "
                       f"(delta {fmt(offset[m])} from fixed-budget 17 vs {len(high)})"),
            ))
        md_pred[P] = dict(status="forecast", B=budget, stats=point, band=band,
                          cross=cross, n_missing=len(missing), n_obs=len(high))

    return rows, lang_rows, md_pred


def self_check(gen_dir, args, rows):
    """At P=66 the proportional budget IS the fixed budget — the trees must agree."""
    if 66 not in args.pool_sizes:
        return None
    langs = list(DEFAULT_LANGS)
    a = pool_stats(load_tree(gen_dir, args.model_abbr, 66, args.method, args.seed, langs),
                   66, FIXED_BUDGET)
    b = pool_stats(load_tree(gen_dir, args.budgetfrac_abbr, 66, args.method, args.seed, langs),
                   66, POOL_BUDGETS[66])
    ok = (a is not None and b is not None
          and a["n_langs"] == b["n_langs"]
          and abs(a["naive_fpr_pct"] - b["naive_fpr_pct"]) < 1e-9
          and abs(a["tau_star"] - b["tau_star"]) < 1e-9)
    print(f"[self-check] P=66 identity: {'PASS' if ok else 'FAIL'} — "
          f"fixed {a['naive_fpr_pct']:.2f}% / tau*={a['tau_star']:.3f} vs "
          f"budgetfrac {b['naive_fpr_pct']:.2f}% / tau*={b['tau_star']:.3f}")
    return ok


def lang_table_lines(lang_rows, pool_sizes, budget_name, metric):
    """Markdown table of one per-language metric, langs x pools, with bands."""
    by = {(r["budget"], r["P"], r["lang"]): r for r in lang_rows}
    pools = [P for P in pool_sizes if any(r["budget"] == budget_name and r["P"] == P
                                          for r in lang_rows)]
    lines = ["| Tier | Lang | " + " | ".join(f"P={P}" for P in pools) + " |",
             "| --- | --- | " + " | ".join("---" for _ in pools) + " |"]
    last_tier = None
    for lang in DEFAULT_LANGS:
        tier = RESOURCE_TIER.get(lang, "")
        cells = []
        for P in pools:
            r = by.get((budget_name, P, lang))
            if r is None or r[metric] == "":
                cells.append("—")
                continue
            v, lo, hi = r[metric], r[f"{metric}_low"], r[f"{metric}_high"]
            cells.append(f"{v:.1f}" if lo == hi else f"{v:.1f} ({lo:.1f}–{hi:.1f})")
        label = tier.capitalize() if tier != last_tier else ""
        last_tier = tier
        lines.append(f"| {label} | `{lang}` | " + " | ".join(cells) + " |")
    return lines


def print_lang_table(lang_rows, pool_sizes):
    for budget_name in ("fixed", "proportional"):
        if not any(r["budget"] == budget_name for r in lang_rows):
            continue
        print(f"\nper-language FPR @ global tau* (%) — {budget_name} budget")
        for line in lang_table_lines(lang_rows, pool_sizes, budget_name,
                                     "fpr_at_global_tau_pct"):
            print("  " + line)


def write_md(path, md_pred, pool_sizes, lang_rows):
    lines = [
        "# Predicted Table 3 — proportional budget (30.3%), all 17 languages",
        "",
        "Generated by `estimate_fpr_budgetfrac.py`. Cells marked *forecast* are",
        "predictions for languages still generating; see the CSV for the method and",
        "the cross-check. Re-run once the sweep lands to replace them with measurements.",
        "",
        "| Pool P | B | FPR @ naive z=2.33 | 95% band | tau* | 95% band | FPR @ tau* | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for P in pool_sizes:
        e = md_pred.get(P)
        if not e:
            continue
        st, band = e["stats"], e["band"]

        def b(m, nd=1, pct=True):
            if band is None:
                return "—"
            lo, hi = band[m]
            return f"{lo:.{nd}f}–{hi:.{nd}f}" + ("%" if pct else "")

        lines.append(
            f"| {P} | {e['B']} | {st['naive_fpr_pct']:.1f}% | {b('naive_fpr_pct')} | "
            f"{st['tau_star']:.2f} | {b('tau_star', 2, False)} | "
            f"{st['fpr_at_tau_pct']:.1f}% | {e['status']} |"
        )

    naive = [md_pred[P]["stats"]["naive_fpr_pct"] for P in pool_sizes if P in md_pred]
    taus = [md_pred[P]["stats"]["tau_star"] for P in pool_sizes if P in md_pred]
    lines += [
        "",
        "**Mechanism check.** " + (
            "Naive FPR is monotonically increasing in P at a proportional budget "
            if all(x < y for x, y in zip(naive, naive[1:])) else
            "Naive FPR is NOT monotone in P at a proportional budget "
        ) + "(" + " → ".join(f"{x:.1f}%" for x in naive) + "), and tau* goes "
        + " → ".join(f"{x:.2f}" for x in taus) + ".",
    ]
    for P in pool_sizes:
        e = md_pred.get(P)
        if e and e["status"] == "forecast":
            lo, hi = e["band"]["naive_fpr_pct"]
            lines.append(
                f"- P={P}: forecast from {e['n_obs']} measured + {e['n_missing']} "
                f"synthesised languages; tier-offset cross-check "
                f"{e['cross']['naive_fpr_pct']:.1f}% / tau* {e['cross']['tau_star']:.2f}. "
                f"Band {lo:.1f}–{hi:.1f}%."
            )
    # The claim is that FPR separates across P once the budget scales with it. If a
    # forecast band reaches the neighbouring pool's value, the separation is not
    # resolved by these data and the camera-ready should not lean on it.
    ordered = [P for P in pool_sizes if P in md_pred]
    for a, b in zip(ordered, ordered[1:]):
        for P, other in ((a, b), (b, a)):
            e = md_pred[P]
            if e["status"] != "forecast":
                continue
            lo, hi = e["band"]["naive_fpr_pct"]
            ref = md_pred[other]["stats"]["naive_fpr_pct"]
            if lo <= ref <= hi:
                lines.append(
                    f"- ⚠️ P={P}'s band ({lo:.1f}–{hi:.1f}%) contains P={other}'s "
                    f"value ({ref:.1f}%) — the two pools are not separated."
                )
            else:
                lines.append(
                    f"- P={P} vs P={other}: separated ({lo:.1f}–{hi:.1f}% vs "
                    f"{ref:.1f}%)."
                )
    for budget_name, title in (("fixed", "fixed budget (B=20)"),
                               ("proportional", "proportional budget (30.3%)")):
        if not any(r["budget"] == budget_name for r in lang_rows):
            continue
        lines += [
            "",
            f"## Per-language FPR at the single global tau* (%) — {title}",
            "",
            "Values in parentheses are the 95% forecast band; a bare number is measured.",
            "",
        ]
        lines += lang_table_lines(lang_rows, pool_sizes, budget_name,
                                  "fpr_at_global_tau_pct")
        lines += [
            "",
            f"### Per-language FPR at the naive z=2.33 cutoff (%) — {title}",
            "",
        ]
        lines += lang_table_lines(lang_rows, pool_sizes, budget_name,
                                  "naive_fpr_lang_pct")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_dir", default="gen")
    ap.add_argument("--model_abbr", default="aya-23-8B",
                    help="tree holding the fixed-budget (B=20) nulls")
    ap.add_argument("--budgetfrac_abbr", default="aya-23-8B-budgetfrac",
                    help="tree holding the proportional-budget nulls")
    ap.add_argument("--method", default="kgw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_sizes", type=int, nargs="+", default=[33, 66, 126])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/fpr_calibration_budgetfrac")
    args = ap.parse_args()

    rng = np.random.default_rng(args.boot_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    rows, lang_rows, md_pred = build_rows(args.gen_dir, args, rng)
    self_check(args.gen_dir, args, rows)

    csv_path = os.path.join(args.out_dir, "expected_17lang_forecast.csv")
    write_csv(rows, csv_path, fieldnames=list(rows[0]))
    lang_csv_path = os.path.join(args.out_dir, "expected_17lang_per_lang.csv")
    write_csv(lang_rows, lang_csv_path, fieldnames=list(lang_rows[0]))
    md_path = os.path.join(args.out_dir, "expected_17lang_forecast.md")
    write_md(md_path, md_pred, args.pool_sizes, lang_rows)

    print()
    hdr = f"{'budget':<13}{'P':>5}{'B':>4}  {'metric':<16}{'expected':>10}{'low':>9}{'high':>9}  status"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['budget']:<13}{r['P']:>5}{r['B']:>4}  {r['metric']:<16}"
              f"{str(r['expected']):>10}{str(r['low']):>9}{str(r['high']):>9}  {r['status']}")
    print_lang_table(lang_rows, args.pool_sizes)
    print(f"\nwrote {csv_path}\nwrote {lang_csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
