#!/bin/bash

# Exit on any error or unset variable
set -e
set -u

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"

# Model configuration
TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE=32

# Model names and abbreviations
MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)
MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

# Settings
WATERMARK_METHODS=("kgw" "xsir")
SEEDS=(0)

# Target languages for BO-enhanced STEAM evaluation
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
    # "ru" # Russian
    "hi" # Hindi
    "ko" # Korean
    # "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    # "vi" # Vietnamese
    # "iw" # Hebrew
    # "uk" # Ukrainian
    # "ta" # Tamil
)

# BO Configuration
N_INITIAL=3          # Number of random initial language samples
MAX_EVALUATIONS=8    # Maximum total language evaluations per text
N_SAMPLES=50         # Number of text samples to evaluate (subset for testing)

# Validation check
if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "❌ Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

echo "BO-Enhanced STEAM Evaluation"

# Main evaluation loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    echo "Processing $MODEL_ABBR"

    for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            # Set output directory
            OUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
            mkdir -p "$OUT_DIR"

            # Set watermark-specific flags
            if [ "$WATERMARK_METHOD" == "kgw" ]; then
                WATERMARK_FLAGS="--watermark_method kgw"
            elif [ "$WATERMARK_METHOD" == "xsir" ]; then
                MAPPING_FILE="$DATA_DIR/mapping/xsir/mapping_${MODEL_ABBR}_en_mc4.json"
                WATERMARK_FLAGS="--watermark_method xsir --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_FILE"
            else
                echo "❌ Unknown watermark method: $WATERMARK_METHOD"
                exit 1
            fi

            if [ ! -f "$OUT_DIR/mc4.en.mod.jsonl" ]; then
                echo "Watermarked data not found: $OUT_DIR/mc4.en.mod.jsonl"
                continue
            fi

            # BO-Enhanced STEAM evaluation for each target language
            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo "$WATERMARK_METHOD $TGT_LANG BO-STEAM"

                TRANSLATED_FILE="$OUT_DIR/mc4.en-${TGT_LANG}.mod.jsonl"
                if [ ! -f "$TRANSLATED_FILE" ]; then
                    echo "Translated data not found: $TRANSLATED_FILE"
                    continue
                fi

                BO_OUTPUT_FILE="$OUT_DIR/mc4.en-${TGT_LANG}.bo_steam_results.jsonl"
                BO_SUMMARY_FILE="$OUT_DIR/mc4.en-${TGT_LANG}.bo_steam_summary.json"

                python3 "$WORK_DIR/evaluate_bo_steam.py" \
                    --base_model "$MODEL_NAME" \
                    --input_file "$TRANSLATED_FILE" \
                    --output_file "$BO_OUTPUT_FILE" \
                    --summary_file "$BO_SUMMARY_FILE" \
                    --target_lang "$TGT_LANG" \
                    --n_initial "$N_INITIAL" \
                    --max_evaluations "$MAX_EVALUATIONS" \
                    --n_samples "$N_SAMPLES" \
                    --seed "$SEED" \
                    $WATERMARK_FLAGS

            done


        done
    done
done

echo "BO-Enhanced STEAM Evaluation Complete"