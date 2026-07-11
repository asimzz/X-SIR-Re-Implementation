#!/bin/bash
#
# Independent-null FPR calibration sweep.
#
# For the FPR-inflation rebuttal (W3): the deployed STEAM-BO detector runs a full
# max-over-search on EVERY suspect text, so the null distribution of its output
# statistic must be characterised by running an INDEPENDENT full BO search on each
# human/null text — NOT by borrowing the pivot chosen for a paired watermarked text
# (which is what run()/_process_human_text does and which understates the null).
#
# This script runs steam_bo_detector.py with --independent_human, producing
#   mc4.{lang}.bo.hum.indep.z_score.jsonl
# for candidate pool sizes P ∈ {33, 66, 126}. Positives (mc4.{lang}.bo.z_score.jsonl)
# are REUSED, not regenerated:
#   - P=126 is the full pool (== base kgw_seed0 run; pool_126/kgw_seed0 is a symlink),
#   - P=33/66 positives come from run_steam_pool_sweep.sh.
# Human null inputs (mc4.en-{lang}.hum.jsonl) always come from the base kgw_seed0 dir.
#
# Budget (MAX_EVALUATIONS=20) and POOL_SEED=42 match run_steam_pool_sweep.sh exactly,
# so the ONLY thing varying across pools is the candidate pool itself, and the null
# pool is byte-identical to the positive pool for the same (lang, P).
#
# Every output file resumes independently (skips files already at >= NUM_TEXTS lines;
# steam_bo_detector.run_null drops any partial trailing line and flushes per text).
#
# Usage:
#   ./scripts/run_steam_fpr_null.sh                # full sweep (all langs, all pools)
#   ./scripts/run_steam_fpr_null.sh fr             # single target language, all pools
#   ./scripts/run_steam_fpr_null.sh fr 33          # single language + single pool (dry run)

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
GEN_DIR="$WORK_DIR/gen"

MODEL_NAME="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
WATERMARK_METHOD="kgw"
SEED=0

TGT_LANGS=(
    # High-resource
    "fr" "de" "it" "es" "pt"
    # Medium-resource
    "pl" "nl" "ru" "hi" "ko" "ja"
    # Low-resource
    "bn" "fa" "vi" "iw" "uk" "ta"
)

# Pool sizes for the sweep. NOTE: 126 == full pool (the target-excluded candidate
# set is 125 entries, so --max_candidate_langs 126 is a documented no-op that
# selects the full pool — identical to the base kgw_seed0 run's pool).
POOL_SIZES=(33 66 126)

# Reproducible subsampling seed — MUST match run_steam_pool_sweep.sh so the null
# candidate pool is identical to the positive candidate pool for each (lang, P).
POOL_SEED=42

NUM_TEXTS=500
N_INITIAL=3
MAX_EVALUATIONS=20

# Optional scoping for debugging / dry runs.
if [[ -n "${1:-}" ]]; then
    TGT_LANGS=("$1")
fi
if [[ -n "${2:-}" ]]; then
    POOL_SIZES=("$2")
fi

# Human null inputs and gamma_lang live in the base run directory.
INPUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
GAMMA_LANG_FILE="$INPUT_DIR/gamma_lang.json"

if [ ! -f "$GAMMA_LANG_FILE" ]; then
    echo "ERROR: gamma_lang file not found: $GAMMA_LANG_FILE"
    exit 1
fi

for POOL_SIZE in "${POOL_SIZES[@]}"; do
    OUTPUT_DIR="$GEN_DIR/$MODEL_ABBR/pool_${POOL_SIZE}/${WATERMARK_METHOD}_seed${SEED}"
    # For pool_126 this is a symlink → ../kgw_seed0; mkdir -p is a no-op then.
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
                echo "Skipping pool=$POOL_SIZE $TGT_LANG (complete: $OUT_LINES/$NUM_TEXTS)"
                continue
            fi
        fi

        echo "=== NULL search pool=$POOL_SIZE $MODEL_ABBR $WATERMARK_METHOD seed=$SEED $TGT_LANG ==="

        python3 "$WORK_DIR/steam_bo_detector.py" \
            --base_model "$MODEL_NAME" \
            --tgt_lang "$TGT_LANG" \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --gamma_lang_file "$GAMMA_LANG_FILE" \
            --n_initial "$N_INITIAL" \
            --max_evaluations "$MAX_EVALUATIONS" \
            --num_texts "$NUM_TEXTS" \
            --max_candidate_langs "$POOL_SIZE" \
            --pool_seed "$POOL_SEED" \
            --independent_human

        echo ""
    done
done

echo "Independent-null sweep complete."
