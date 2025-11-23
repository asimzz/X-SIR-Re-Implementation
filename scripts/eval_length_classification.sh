#!/bin/bash

set -e
set -u

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
    # "fr" # French
    "de" # German
    # "it" # Italian
    "es" # Spanish
    # "pt" # Portuguese
    # Medium-resource languages
    # "pl" # Polish
    # "nl" # Dutch
    "ru" # Russian
    "hi" # Hindi
    # "ko" # Korean
    "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    # "vi" # Vietnamese
    "iw" # Hebrew
    # "uk" # Ukrainian
    # "ta" # Tamil
    )
SEEDS=(0)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

# Create results directory if it doesn't exist
RESULTS_DIR=$WORK_DIR/results_length_classification
mkdir -p $RESULTS_DIR

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do

            WATERMARK_DIR=$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}

            echo "======================================="
            echo "Model: $MODEL_NAME"
            echo "Watermark Method: $WATERMARK_METHOD (seed=$SEED)"
            echo "======================================="

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo ""
                echo ">>> Processing Target Language: $TGT_LANG <<<"
                echo ""

                # Check if the required binned files exist
                SHORT_FILE="$WATERMARK_DIR/mc4.en-${TGT_LANG}.hum.z_score.short.z_score.jsonl"
                if [ ! -f "$SHORT_FILE" ]; then
                    echo "Warning: Binned files not found for $TGT_LANG, skipping..."
                    echo "Expected file: $SHORT_FILE"
                    continue
                fi

                echo "Evaluating Length Classification for $MODEL_NAME $WATERMARK_METHOD (seed=$SEED) -> $TGT_LANG"

                # Run length classification evaluation
                python3 $WORK_DIR/evaluate_length_classification.py \
                    --tgt_lang "$TGT_LANG" \
                    --base_wm_dir "$WATERMARK_DIR" \
                    --roc_curve "$RESULTS_DIR/${MODEL_ABBR}_${WATERMARK_METHOD}_seed${SEED}_${TGT_LANG}_length_roc.png"

                echo ""
            done

            echo "======================================="
            echo "Completed evaluation for $MODEL_NAME $WATERMARK_METHOD (seed=$SEED)"
            echo "Results saved to: $RESULTS_DIR"
            echo "======================================="
        done
    done
done

echo ""
echo "🎉 All length classification evaluations completed!"
echo "📊 Results and ROC curves saved to: $RESULTS_DIR"
echo ""
echo "Generated files:"
find $RESULTS_DIR -name "*.png" -o -name "*.txt" | sort