#!/bin/bash
#
# Pool-size FPR calibration sweep.
#
# Runs STEAM-BO with candidate pool sizes N ∈ {33, 66, 133}, producing parallel
# output trees under .../pool_${N}/ to keep the rebuttal data isolated from the
# production sweep. Inputs (watermarked .mod / .hum jsonl) come from the
# existing kgw_seed${SEED} run; γ_lang is reused as-is.
#
# Usage:
#   ./scripts/run_steam_pool_sweep.sh             # full sweep (all langs)
#   ./scripts/run_steam_pool_sweep.sh fr          # single target language

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
GEN_DIR="$WORK_DIR/gen"

MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)
MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

WATERMARK_METHODS=("kgw")
SEEDS=(0)

TGT_LANGS=(
    # High-resource
    "fr" "de" "it" "es" "pt"
    # Medium-resource
    "pl" "nl" "ru" "hi" "ko" "ja"
    # Low-resource
    "bn" "fa" "vi" "iw" "uk" "ta"
)

# Pool sizes for the sweep
POOL_SIZES=(33 66 133)

# Reproducible subsampling: same seed across all (method, seed, lang) so the
# 33-language subset is identical for every run, and likewise for 66.
POOL_SEED=42

# How many texts to process per language (matches run_steam_bo.sh default)
NUM_TEXTS=500

# Allow scoping to a single target language for debugging
if [[ -n "${1:-}" ]]; then
    TGT_LANGS=("$1")
fi

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for POOL_SIZE in "${POOL_SIZES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do

                INPUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
                OUTPUT_DIR="$GEN_DIR/$MODEL_ABBR/pool_${POOL_SIZE}/${WATERMARK_METHOD}_seed${SEED}"
                GAMMA_LANG_FILE="$GEN_DIR/$MODEL_ABBR/kgw_seed${SEED}/gamma_lang.json"

                mkdir -p "$OUTPUT_DIR"

                for TGT_LANG in "${TGT_LANGS[@]}"; do
                    MOD_FILE="$INPUT_DIR/mc4.en-${TGT_LANG}.mod.jsonl"
                    HUM_FILE="$INPUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl"

                    if [ ! -f "$MOD_FILE" ] || [ ! -f "$HUM_FILE" ]; then
                        echo "⚠️  Missing inputs for $MODEL_ABBR $WATERMARK_METHOD seed=$SEED $TGT_LANG — skipping"
                        continue
                    fi

                    # Skip if already complete
                    MOD_OUT="$OUTPUT_DIR/mc4.${TGT_LANG}.bo.z_score.jsonl"
                    HUM_OUT="$OUTPUT_DIR/mc4.${TGT_LANG}.bo.hum.z_score.jsonl"
                    if [[ -f "$MOD_OUT" && -f "$HUM_OUT" ]]; then
                        MOD_LINES=$(wc -l < "$MOD_OUT" | tr -d ' ')
                        HUM_LINES=$(wc -l < "$HUM_OUT" | tr -d ' ')
                        if [[ "$MOD_LINES" -ge "$NUM_TEXTS" && "$HUM_LINES" -ge "$NUM_TEXTS" ]]; then
                            echo "Skipping pool=$POOL_SIZE $MODEL_ABBR $WATERMARK_METHOD seed=$SEED $TGT_LANG (complete: $MOD_LINES/$NUM_TEXTS)"
                            continue
                        fi
                    fi

                    echo "=== pool=$POOL_SIZE $MODEL_ABBR $WATERMARK_METHOD seed=$SEED $TGT_LANG ==="

                    python3 "$WORK_DIR/steam_bo_detector.py" \
                        --base_model "$MODEL_NAME" \
                        --tgt_lang "$TGT_LANG" \
                        --input_dir "$INPUT_DIR" \
                        --output_dir "$OUTPUT_DIR" \
                        --gamma_lang_file "$GAMMA_LANG_FILE" \
                        --num_texts "$NUM_TEXTS" \
                        --max_candidate_langs "$POOL_SIZE" \
                        --pool_seed "$POOL_SEED"

                    echo ""
                done
            done
        done
    done
done
