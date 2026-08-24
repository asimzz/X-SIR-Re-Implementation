# Camera-ready W3 — empirical FPR calibration of the max-over-search statistic

Deliverable for reviewer actions **3.1 / C1.1 / 2.1** (empirical calibration study with
per-language FPR) and **3.2 / 2.2** (explicit statistical control of the max statistic).
All numbers below are over **all 17 target languages**, 500 held-out human texts each
(8500 null max-scores per pool), aya-23-8B / KGW / seed 0.

Reproduce with:

```bash
python3 analyze_fpr_calibration.py --model_abbr aya-23-8B \
  --pool_sizes 33 66 126 --no_tpr --require_langs 17 --emit_latex \
  --out_dir results/fpr_calibration_fixed20
```

LaTeX for both tables: `results/fpr_calibration_fixed20/fpr_calibration_tables_*.tex`.

---

## Table 1 — Calibration at the fixed BO budget used throughout the paper (B = 20)

| Pool P | B | FPR @ naive z=2.33 | z_Bonf | FPR @ z_Bonf | τ\* | **FPR @ τ\*** |
| --- | --- | --- | --- | --- | --- | --- |
| 33 | 20 | 29.2% | 3.29 | 5.4% | 3.94 | **1.1%** |
| 66 | 20 | 25.8% | 3.29 | 4.4% | 3.87 | **1.0%** |
| 126 | 20 | 25.8% | 3.29 | 4.2% | 3.90 | **1.0%** |

Three things to say about this table:

1. **The inflation is real and large.** At the naive single-test 1% threshold the max
   statistic false-positives on ~26-29% of human texts. This is the effect W3 warns of,
   and it is why every TPR@FPR=1% in the paper is thresholded on the null of the full
   max statistic rather than on a single-detection null.
2. **It does not grow with the pool.** FPR at the naive cutoff is flat-to-decreasing in
   P (29.2 → 25.8 → 25.8%), and τ\* is flat at 3.87-3.94. A threshold calibrated once
   transfers as language coverage grows.
3. **Calibration works.** The empirically calibrated τ\* achieves 1.0-1.1% on a disjoint
   verification split — the target, not merely "controlled".

Stability: across split seeds 0-4, τ\* ∈ [3.81, 4.02] and achieved FPR ∈ [0.6%, 1.4%],
flat in P at every seed.

## Table 2 — Per-language FPR at the *single global* τ\*

FPR (%) on the held-out verification split, at the one global threshold of Table 1.

| Tier | Lang | P=33 | P=66 | P=126 |
| --- | --- | --- | --- | --- |
| High | fr | 0.8 | 0.4 | 1.2 |
| | de | 0.8 | 0.8 | 1.6 |
| | it | 1.2 | 1.2 | 1.2 |
| | es | 0.8 | 0.8 | 0.4 |
| | pt | 0.4 | 1.6 | 1.6 |
| | **mean** | **0.8** | **1.0** | **1.2** |
| Medium | pl | 2.4 | 2.4 | 1.2 |
| | nl | 0.4 | 1.2 | 0.4 |
| | ru | 1.6 | 1.6 | 0.4 |
| | hi | 0.8 | 1.6 | 1.2 |
| | ko | 0.8 | 1.2 | 1.2 |
| | ja | 0.8 | 1.2 | 1.2 |
| | **mean** | **1.1** | **1.5** | **0.9** |
| Low | bn | 1.2 | 0.0 | 1.6 |
| | fa | 1.2 | 1.2 | 0.8 |
| | vi | 1.6 | 0.4 | 0.4 |
| | iw | 1.6 | 0.4 | 0.0 |
| | uk | 2.0 | 0.8 | 1.6 |
| | ta | 0.8 | 0.4 | 0.4 |
| | **mean** | **1.4** | **0.5** | **0.8** |

**No tier absorbs a disproportionate share of the error budget** — tier means span
0.5-1.5% against a 1% target, with no systematic ordering by resource level. The worst
single cell is 2.4% (`pl`), which on 250 verification texts is 6 false positives against
an expected 2.5 — about 2σ of binomial noise, not a structural failure. This is the
direct answer to the Tamil-style concern: `ta` is at or below target at every pool
(0.8 / 0.4 / 0.4%). It is what the language-specific null correction γ_ℓ buys.

### Per-language thresholds are *worse* than the global one

We also fitted a per-language τ\*_ℓ (99th percentile of each language's own calibration
half) as a robustness check. It performs **worse**: at P=126 it gives 5.2% FPR for `de`
and 3.2% for `bn`, against 1.6% for both at the global τ\*. The reason is estimator
variance — a 99th percentile from 250 points rests on ~2.5 order statistics, whereas the
pooled global τ\* is estimated from 4250. Pooling across languages is therefore the
right design, and it is defensible precisely *because* γ_ℓ has already removed the
per-language location bias. Worth one sentence in the paper; it pre-empts "why not a
threshold per language?".

## Table 3 — Control: budget proportional to pool size (30.3%)

⚠️ **Partial — P=33 and P=126 currently cover only the 5 high-resource languages.**
Pending `scripts/run_steam_fpr_budget_all17.sh` (12 languages × 2 pools). P=66 is the
30.3% anchor and is the 17-language fixed-budget run, reused.

| Pool P | B | FPR @ naive z=2.33 | τ\* | langs |
| --- | --- | --- | --- | --- |
| 33 | 10 | 17.7% | 3.78 | 5 |
| 66 | 20 | 25.8% | 3.87 | 17 |
| 126 | 38 | 41.1% | 4.13 | 5 |

The mechanism claim holds: when the budget scales with the pool, FPR at the naive cutoff
**does** rise (17.7 → 25.8 → 41.1%) and τ\* rises with it (3.78 → 3.87 → 4.13).
Comparing Table 1 with Table 3 is the whole argument: **search inflation is governed by
the evaluation budget B, not by the pool size P.** Since every experiment in the paper
uses B = 20 — only 16% of the pool at P = 126 — the multiple-testing cost is bounded and
stable however far language support grows.

---

## Statistical control of the max statistic (§ for the paper)

The statistic is the maximum over the **B = min(budget, P) = 20** candidates the BO
search actually evaluates, not over all P candidates in the pool. This is the pivotal
point, and it should be stated in terms of B throughout:

| Quantity | P=33 | P=66 | P=126 |
| --- | --- | --- | --- |
| B (hypotheses tested) | 20 | 20 | 20 |
| √(2 ln B) | 2.45 | 2.45 | 2.45 |
| Gumbel location Φ⁻¹(1 − 1/B) | 1.64 | 1.64 | 1.64 |
| Bonferroni z = Φ⁻¹(1 − α/B) | 3.29 | 3.29 | 3.29 |
| **Empirical τ\*** | **3.94** | **3.87** | **3.90** |
| p_eff = α / SF(τ\*) | 247 | 187 | 206 |

Under an iid standard-Gaussian per-candidate null, the maximum of B scores concentrates
at √(2 ln B) with Gumbel location Φ⁻¹(1 − 1/B). Both depend on **B alone**, so both are
constant in P at a fixed budget — which is exactly why a single calibrated threshold
transfers as the pool grows, and why the empirical τ\* in Table 1 is flat. P enters only
through `min(budget, P)`, which does not bind at any pool we evaluate.

### Two corrections to the rebuttal text

**(a) Use B, not P.** Our first response argued the null max concentrates around
√(2 log P), "growing from ≈2.6 at P=33 to ≈3.1 at P=126". That is the wrong count and it
contradicts our own follow-up, which showed τ\* is flat in P. The correct statement is
√(2 ln B) ≈ 2.45, constant in P. Write the B-based version.

**(b) Drop the "positive correlation shrinks the inflation" claim — the data contradicts
it.** We asserted that positive correlation between related back-translation candidates
makes the true inflation smaller than the multiple-testing bound. Measured, the opposite
holds: p_eff ≈ 187-247, an order of magnitude **above** B = 20, and Bonferroni over B
leaves 4.2-5.4% FPR rather than ≤1%. The per-candidate γ_ℓ-corrected null is therefore
**heavier-tailed than Gaussian**, so a Bonferroni/Šidák correction over the budget is
*not* conservative here.

This is a stronger argument for the paper than the one we made. It says the empirical
calibration in Table 1 is not a convenience that happens to be tighter than theory — it
is *necessary*, because no Gaussian-tail correction over B would hold FPR at 1% on this
statistic. It also explains the headline number cleanly: the naive cutoff fails by ~26
percentage points not because 20 tests is a lot, but because the per-candidate null has
a heavy right tail that only a distribution-free empirical quantile captures.

Suggested framing: *"Because the per-candidate null is heavier-tailed than Gaussian
(p_eff ≫ B), we set the decision threshold as an empirical quantile of the full
max-over-search null rather than by an analytic multiple-testing correction. The
correction is bounded by the search budget B, not the pool size P, so it does not grow
as language coverage scales."*

---

## Provenance and scope

- **FPR-only.** No TPR at τ\* is reported. Every committed positive
  (`mc4.{lang}.bo.z_score.jsonl`) was generated on the buggy 84-pivot pool, and at
  P=33/66 the positives' subsample was drawn from 84 candidates while the nulls' was
  drawn from 125 — they do not search the same languages. `--no_tpr` enforces this;
  none of the three reviewer actions asked for TPR.
- **Nulls are clean.** All 95 `*.bo.hum.indep.z_score.jsonl` files verified: 125-pool
  provenance, exactly 500 records, zero `None` z-scores.
- **Independent-search nulls.** These are `--independent_human` outputs, where each null
  text runs its own full BO search. The older paired-pivot nulls borrow the watermarked
  text's pivot and understate the null.
- **Disjointness.** γ_ℓ is fit on `mc4.{lang}.val.jsonl`, disjoint from the mc4 human
  test texts used here; the calibration and verification halves are disjoint from each
  other by a seeded per-language permutation.
- **Split mode.** The mc4 corpus is ordered — within every language the first 250 texts
  have a heavier upper tail (at P=126, p99 4.03 vs 3.74; max 7.74 vs 4.83). The original
  index split was therefore not exchangeable and landed achieved FPR at ~0.4% instead of
  1%. The camera-ready uses `--split_mode random` (default). `--split_mode index`
  reproduces the July rebuttal exactly.

## Regression against the posted rebuttal

`--langs fr de it es pt --split_mode index` reproduces posted Table R2 **bit-for-bit**:

| P | Posted FPR / τ\* | Reproduced |
| --- | --- | --- |
| 33 | 27.6% / 4.36 | 27.6% / 4.36 |
| 66 | 25.9% / 4.05 | 25.9% / 4.05 |
| 126 | 25.2% / 4.07 | 25.2% / 4.07 |

So the camera-ready numbers differ from the posted ones only through two deliberate,
documented changes — 17 languages instead of 5, and an exchangeable split — and the
qualitative conclusion (FPR stable in P, threshold transfers) is unchanged.
Output: `results/fpr_calibration_high5/`.
