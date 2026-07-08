#!/bin/bash

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"

# Model configuration
TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"

# Model and settings
MODEL_NAME="meta-llama/Llama-3.2-1B"
MODEL_ABBR="llama-3.2-1B"
WATERMARK_METHOD="kgw"
SEED=0

# Target languages for per-text evaluation
TARGET_LANGS=(
    "fr" # French
    "de" # German
    "it" # Italian
    "es" # Spanish
    "pt" # Portuguese
    "pl" # Polish
    "nl" # Dutch
    "ru" # Russian
    "hi" # Hindi
    "ko" # Korean
    "ja" # Japanese
    "bn" # Bengali
    "fa" # Persian
    "vi" # Vietnamese
    "iw" # Hebrew
    "uk" # Ukrainian
    "ta" # Tamil
)

# STEAM per-text settings
N_INITIAL=3
MAX_EVALUATIONS=8

# Set watermark flags
if [ "$WATERMARK_METHOD" == "kgw" ]; then
    WATERMARK_FLAGS="--watermark_method kgw"
elif [ "$WATERMARK_METHOD" == "xsir" ]; then
    MAPPING_FILE="$DATA_DIR/mapping/xsir/mapping_${MODEL_ABBR}_en_mc4.json"
    WATERMARK_FLAGS="--watermark_method xsir --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_FILE"
fi

echo "🚀 Starting STEAM per-text workflow for ${#TARGET_LANGS[@]} languages"

# Main loop for each target language
for TARGET_LANG in "${TARGET_LANGS[@]}"; do
    echo ""
    echo "🎯 Processing target language: $TARGET_LANG"

    OUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
    RESULTS_DIR="$OUT_DIR/steam_per_text"
    mkdir -p "$RESULTS_DIR"

    # Check if required input files exist
    WM_INPUT="$OUT_DIR/mc4.en-${TARGET_LANG}.mod.jsonl"
    HM_INPUT="$OUT_DIR/mc4.en-${TARGET_LANG}.hum.jsonl"
    VAL_INPUT="$OUT_DIR/mc4.en-${TARGET_LANG}.val.jsonl"

    if [ ! -f "$WM_INPUT" ] || [ ! -f "$HM_INPUT" ] || [ ! -f "$VAL_INPUT" ]; then
        echo "⚠️  Missing input files for $TARGET_LANG:"
        echo "   Watermarked: $WM_INPUT"
        echo "   Human: $HM_INPUT"
        echo "   Validation: $VAL_INPUT"
        echo "   Skipping $TARGET_LANG"
        continue
    fi

    echo "📊 Running STEAM per-text for watermarked texts..."

    # Run STEAM per-text for watermarked texts
    cd "$WORK_DIR"
    python3 steam_per_text.py \
        --input_file "$WM_INPUT" \
        --output_file "$RESULTS_DIR/steam_${TARGET_LANG}_watermarked.json" \
        --target_lang "$TARGET_LANG" \
        --base_model "$MODEL_NAME" \
        --seed "$SEED" \
        --n_initial "$N_INITIAL" \
        --max_evaluations "$MAX_EVALUATIONS" \
        --val_file "$VAL_INPUT" \
        --is_watermarked \
        $WATERMARK_FLAGS

    echo "📊 Running STEAM per-text for human texts..."

    # Run STEAM per-text for human texts (no --is_watermarked flag)
    python3 steam_per_text.py \
        --input_file "$HM_INPUT" \
        --output_file "$RESULTS_DIR/steam_${TARGET_LANG}_human.json" \
        --target_lang "$TARGET_LANG" \
        --base_model "$MODEL_NAME" \
        --seed "$SEED" \
        --n_initial "$N_INITIAL" \
        --max_evaluations "$MAX_EVALUATIONS" \
        --val_file "$VAL_INPUT" \
        $WATERMARK_FLAGS

    echo "📈 Evaluating STEAM per-text results..."

    # Evaluate results
    python3 evaluate_steam_per_text.py \
        --watermark_results "$RESULTS_DIR/steam_${TARGET_LANG}_watermarked.json" \
        --human_results "$RESULTS_DIR/steam_${TARGET_LANG}_human.json" \
        --roc_curve "$RESULTS_DIR/steam_${TARGET_LANG}_roc.txt"

    echo "✅ Completed STEAM per-text for $TARGET_LANG"
done

echo ""
echo "🎉 All target languages completed!"
echo "Results saved in: $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}/steam_per_text/"