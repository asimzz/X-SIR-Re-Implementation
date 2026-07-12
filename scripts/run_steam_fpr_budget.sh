#!/bin/bash
#
# Constant-budget-fraction FPR calibration sweep (supervisor follow-up to W3).
#
# The independent-null sweep (scripts/run_steam_fpr_null.sh) holds the BO budget
# FIXED at 20 evaluations across pools P ∈ {33, 66, 126}. That is not
# apples-to-apples: 20 evals is 60% of the P=33 pool but only 16% of P=126. Here
# we instead hold the budget at a CONSTANT FRACTION of the pool (30.3% == 20/66):
#
#     P=33  → budget 10   (round(0.303 * 33))
#     P=66  → budget 20    (== the existing null sweep; REUSED via symlink, not run)
#     P=126 → budget 38   (round(0.303 * 125), the target-excluded full pool)
#
# so the search intensity (fraction of pool explored) is equalised across pools.
# Goal: show the calibrated τ* keeps achieved FPR ≈1% even when the budget scales
# up with the pool — i.e. STEAM-BO's FPR is not inflated by pool size per se.
#
# Runs steam_bo_detector.py with --independent_human, producing
#   mc4.{lang}.bo.hum.indep.z_score.jsonl
# in a SIBLING model_abbr tree (aya-23-8B-budgetfrac) so analyze_fpr_calibration.py
# reads it unchanged via --model_abbr aya-23-8B-budgetfrac. Nulls only (no positives).
#
# The candidate pool subsampling is budget-independent (fixed by --pool_seed +
# --max_candidate_langs), so budget-10@P33 searches the SAME 33-lang subset as the
# fixed-budget-20 null sweep — a clean, controlled comparison.
#
# Every output file resumes independently (skips files already at >= NUM_TEXTS lines;
# steam_bo_detector.run_null drops any partial trailing line and flushes per text).
#
# Usage:
#   ./scripts/run_steam_fpr_budget.sh                # full sweep (5 langs, pools 33 & 126)
#   ./scripts/run_steam_fpr_budget.sh fr             # single target language
#   ./scripts/run_steam_fpr_budget.sh fr 33          # single language + single pool

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
GEN_DIR="$WORK_DIR/gen"

MODEL_NAME="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
OUT_ABBR="aya-23-8B-budgetfrac"     # sibling tree for this experiment's outputs
WATERMARK_METHOD="kgw"
SEED=0

# Only the 5 high-resource null languages (the `high5` set), matching the fixed-budget
# reference experiment (results/fpr_calibration_high5).
TGT_LANGS=(
    "fr" "de" "it" "es" "pt"
)

# (pool_size:budget) pairs on the constant-30.3% line. P=66/budget-20 is the
# existing null sweep and is REUSED via symlink below, so it is NOT listed here.
POOL_BUDGETS=(
    "33:10"
    "126:38"
)

# Reproducible subsampling seed — MUST match run_steam_fpr_null.sh / run_steam_pool_sweep.sh
# so the candidate pool for a given P is identical (budget only changes how many of
# that pool are evaluated).
POOL_SEED=42

NUM_TEXTS=500
N_INITIAL=3

# Optional scoping for debugging / dry runs.
if [[ -n "${1:-}" ]]; then
    TGT_LANGS=("$1")
fi
if [[ -n "${2:-}" ]]; then
    # Keep only the (pool:budget) pair whose pool matches $2.
    _filtered=()
    for pb in "${POOL_BUDGETS[@]}"; do
        if [[ "${pb%%:*}" == "$2" ]]; then
            _filtered+=("$pb")
        fi
    done
    POOL_BUDGETS=("${_filtered[@]}")
fi

# Human null inputs and gamma_lang live in the base run directory.
INPUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
GAMMA_LANG_FILE="$INPUT_DIR/gamma_lang.json"

if [ ! -f "$GAMMA_LANG_FILE" ]; then
    echo "ERROR: gamma_lang file not found: $GAMMA_LANG_FILE"
    exit 1
fi

# Reuse the fixed-budget-20 P=66 nulls (already the 30.3% anchor) via a symlink so
# the analysis picks them up without recomputation.
BUDGETFRAC_ROOT="$GEN_DIR/$OUT_ABBR"
mkdir -p "$BUDGETFRAC_ROOT"
if [ ! -e "$BUDGETFRAC_ROOT/pool_66" ]; then
    ln -s "../$MODEL_ABBR/pool_66" "$BUDGETFRAC_ROOT/pool_66"
    echo "Linked $BUDGETFRAC_ROOT/pool_66 → ../$MODEL_ABBR/pool_66 (reused budget-20 anchor)"
fi

for PB in "${POOL_BUDGETS[@]}"; do
    POOL_SIZE="${PB%%:*}"
    BUDGET="${PB##*:}"
    OUTPUT_DIR="$BUDGETFRAC_ROOT/pool_${POOL_SIZE}/${WATERMARK_METHOD}_seed${SEED}"
    mkdir -p "$OUTPUT_DIR"

    for TGT_LANG in "${TGT_LANGS[@]}"; do
        HUM_IN="$INPUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl"
        if [ ! -f "$HUM_IN" ]; then
            echo "⚠️  Missing null input for $TGT_LANG ($HUM_IN) — skipping"
            continue
        fi

        OUT="$OUTPUT_DIR/mc4.${TGT_LANG}.bo.hum.indep.z_score.jsonl"
        if [ -f "$OUT" ]; then
            OUT_LINES=$(wc -l < "$OUT" | tr -d ' ')
            if [[ "$OUT_LINES" -ge "$NUM_TEXTS" ]]; then
                echo "Skipping pool=$POOL_SIZE budget=$BUDGET $TGT_LANG (complete: $OUT_LINES/$NUM_TEXTS)"
                continue
            fi
        fi

        echo "=== NULL search pool=$POOL_SIZE budget=$BUDGET $OUT_ABBR $WATERMARK_METHOD seed=$SEED $TGT_LANG ==="

        python3 "$WORK_DIR/steam_bo_detector.py" \
            --base_model "$MODEL_NAME" \
            --tgt_lang "$TGT_LANG" \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --gamma_lang_file "$GAMMA_LANG_FILE" \
            --n_initial "$N_INITIAL" \
            --max_evaluations "$BUDGET" \
            --num_texts "$NUM_TEXTS" \
            --max_candidate_langs "$POOL_SIZE" \
            --pool_seed "$POOL_SEED" \
            --independent_human

        echo ""
    done
done

echo "Constant-budget-fraction null sweep complete."
