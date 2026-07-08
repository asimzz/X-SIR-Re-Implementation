#!/bin/bash

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"

# Model and settings
MODEL_NAME="meta-llama/Llama-3.2-1B"
MODEL_ABBR="llama-3.2-1B"
WATERMARK_METHOD="kgw"
SEED=0

# Target languages for simple per-text evaluation
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

echo "🚀 Starting simple STEAM per-text workflow for ${#TARGET_LANGS[@]} languages"

# Main loop for each target language
for TARGET_LANG in "${TARGET_LANGS[@]}"; do
    echo ""
    echo "🎯 Processing target language: $TARGET_LANG"

    BASE_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
    RESULTS_DIR="$BASE_DIR/steam_simple_per_text"
    mkdir -p "$RESULTS_DIR"

    # Check if required input files exist
    MAIN_WM_FILE="$BASE_DIR/mc4.en-${TARGET_LANG}.mod.z_score.jsonl"
    MAIN_HM_FILE="$BASE_DIR/mc4.en-${TARGET_LANG}.hum.z_score.jsonl"

    if [ ! -f "$MAIN_WM_FILE" ] || [ ! -f "$MAIN_HM_FILE" ]; then
        echo "⚠️  Missing main files for $TARGET_LANG:"
        echo "   Watermarked: $MAIN_WM_FILE"
        echo "   Human: $MAIN_HM_FILE"
        echo "   Skipping $TARGET_LANG"
        continue
    fi

    # Count available intermediate language files
    AVAILABLE_LANGS=0
    for LANG in fr de it es pt pl nl ru hi ko ja bn fa vi iw uk ta; do
        if [ "$LANG" != "$TARGET_LANG" ]; then
            WM_INTER_FILE="$BASE_DIR/mc4.${TARGET_LANG}-${LANG}-back.mod.z_score.jsonl"
            HM_INTER_FILE="$BASE_DIR/mc4.${TARGET_LANG}-${LANG}-back.hum.z_score.jsonl"
            VAL_INTER_FILE="$BASE_DIR/mc4.${TARGET_LANG}-${LANG}-back.val.z_score.jsonl"

            if [ -f "$WM_INTER_FILE" ] && [ -f "$HM_INTER_FILE" ] && [ -f "$VAL_INTER_FILE" ]; then
                ((AVAILABLE_LANGS++))
            fi
        fi
    done

    echo "📊 Found $AVAILABLE_LANGS intermediate language files for $TARGET_LANG"

    if [ "$AVAILABLE_LANGS" -eq 0 ]; then
        echo "⚠️  No intermediate language files found for $TARGET_LANG. Skipping."
        continue
    fi

    echo "📈 Running simple STEAM per-text evaluation..."

    # Run simple per-text evaluation
    cd "$WORK_DIR"
    python3 steam_simple_per_text.py \
        --base_dir "$BASE_DIR" \
        --tgt_lang "$TARGET_LANG" \
        --roc_curve "$RESULTS_DIR/steam_simple_${TARGET_LANG}_roc.txt"

    echo "✅ Completed simple STEAM per-text for $TARGET_LANG"
done

echo ""
echo "🎉 All target languages completed!"
echo "Results saved in: $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}/steam_simple_per_text/"