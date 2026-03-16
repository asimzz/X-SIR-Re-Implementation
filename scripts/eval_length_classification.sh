#!/bin/bash

set -e
set -u

# Length classification evaluation using STEAM BO output files.
# Splits texts into short/medium/long by percentile-based token length
# and evaluates watermark detection per bin.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
GEN_DIR=$WORK_DIR/gen

MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

WATERMARK_METHODS=("kgw")
TGT_LANGS=(
    # High-resource languages
    "fr" # French
    "de" # German
    "it" # Italian
    "es" # Spanish
    "pt" # Portuguese
    # Medium-resource languages
    "pl" # Polish
    "nl" # Dutch
    "ru" # Russian
    "hi" # Hindi
    "ko" # Korean
    "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    "vi" # Vietnamese
    "iw" # Hebrew
    "uk" # Ukrainian
    "ta" # Tamil
    )
SEEDS=(0)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

# Create results directory if it doesn't exist
RESULTS_DIR=$WORK_DIR/results_length_classification
mkdir -p $RESULTS_DIR

echo "=== Length Classification (STEAM BO, percentile-based) ==="
echo ""

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do

            WATERMARK_DIR=$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}

            echo "--- $MODEL_ABBR / $WATERMARK_METHOD (seed=$SEED) ---"

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                WM_FILE="$WATERMARK_DIR/mc4.${TGT_LANG}.bo.z_score.jsonl"
                HUM_FILE="$WATERMARK_DIR/mc4.${TGT_LANG}.bo.hum.z_score.jsonl"

                if [ ! -f "$WM_FILE" ] || [ ! -f "$HUM_FILE" ]; then
                    echo "Skipping $TGT_LANG (BO files not found)"
                    continue
                fi

                python3 $WORK_DIR/evaluate_length_classification.py \
                    --tgt_lang "$TGT_LANG" \
                    --base_wm_dir "$WATERMARK_DIR" \
                    --tokenizer "$MODEL_NAME" \
                    --output_csv "$RESULTS_DIR/${MODEL_ABBR}_${WATERMARK_METHOD}_seed${SEED}_${TGT_LANG}_length.csv"

            done
        done
    done
done

echo ""
echo "Done. Results in: $RESULTS_DIR"
find $RESULTS_DIR -name "*.csv" | sort
