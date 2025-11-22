#!/bin/bash

# Exit on any error or unset variable
set -e
set -u

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"

# Model names and abbreviations (same as generate script)
MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)
MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

# Settings (same as generate script)
WATERMARK_METHODS=("kgw")
SEEDS=(0)
TGT_LANGS=("bn" "de" "es" "fa" "fr" "hi" "it" "iw" "ja" "ko" "nl" "pl" "pt" "ru" "ta" "uk" "vi")
ORG_LANGS=("en" "bn" "de" "es" "fa" "fr" "hi" "it" "iw" "ja" "ko" "nl" "pl" "pt" "ru" "ta" "uk" "vi")

# Function to print usage
usage() {
    echo "Usage: $0 [--model MODEL_ABBR] [--seed SEED] [--watermark WATERMARK_METHOD] [--target_lang TARGET_LANG]"
    echo ""
    echo "Options:"
    echo "  --model         Model abbreviation (default: all models)"
    echo "  --seed          Seed value (default: all seeds)"
    echo "  --watermark     Watermark method (default: all methods)"
    echo "  --target_lang   Target language (default: all languages)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Calculate for all combinations"
    echo "  $0 --model aya-23-8B --target_lang ar # Calculate for specific model and language"
    exit 1
}

# Parse command line arguments
FILTER_MODEL=""
FILTER_SEED=""
FILTER_WATERMARK=""
FILTER_TARGET_LANG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --seed)
            FILTER_SEED="$2"
            shift 2
            ;;
        --watermark)
            FILTER_WATERMARK="$2"
            shift 2
            ;;
        --target_lang)
            FILTER_TARGET_LANG="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate model list lengths
if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "❌ MODEL_NAMES and MODEL_ABBRS length mismatch."
    exit 1
fi

echo "🧮 Back-Translation Cost Calculator"
echo "=================================="

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    # Filter by model if specified
    if [[ -n "$FILTER_MODEL" && "$MODEL_ABBR" != "$FILTER_MODEL" ]]; then
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        # Filter by seed if specified
        if [[ -n "$FILTER_SEED" && "$SEED" != "$FILTER_SEED" ]]; then
            continue
        fi

        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            # Filter by watermark method if specified
            if [[ -n "$FILTER_WATERMARK" && "$WATERMARK_METHOD" != "$FILTER_WATERMARK" ]]; then
                continue
            fi

            echo ""
            echo "📊 Model: $MODEL_NAME (seed=$SEED, watermark=$WATERMARK_METHOD)"

            BASE_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"

            if [ ! -d "$BASE_DIR" ]; then
                echo "⚠️  Directory not found: $BASE_DIR"
                continue
            fi

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                # Filter by target language if specified
                if [[ -n "$FILTER_TARGET_LANG" && "$TGT_LANG" != "$FILTER_TARGET_LANG" ]]; then
                    continue
                fi

                INPUT_FILE="$BASE_DIR/mc4.en-${TGT_LANG}.mod.jsonl"

                if [ ! -f "$INPUT_FILE" ]; then
                    echo "⚠️  Input file not found: $INPUT_FILE"
                    continue
                fi

                echo ""
                echo "🌍 Calculating costs for target language: $TGT_LANG"

                # Output file for results - save in new costs_17lang folder to avoid overwriting
                COST_DIR="$DATA_DIR/costs_17lang/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
                mkdir -p "$COST_DIR"
                OUTPUT_FILE="$COST_DIR/mc4.en-${TGT_LANG}_back_translation_cost.json"

                # Run the Python cost calculator
                python3 "$WORK_DIR/calculate_back_translation_cost.py" \
                    --input_file "$INPUT_FILE" \
                    --base_model "$MODEL_NAME" \
                    --output_file "$OUTPUT_FILE"

                echo "✅ Results saved to: $OUTPUT_FILE"
            done
        done
    done
done

echo ""
echo "🎯 Cost calculation complete!"
echo ""
echo "💡 To view summary of all results:"
echo "   find $DATA_DIR/costs_17lang -name '*_back_translation_cost.json' | head -5"
echo ""
echo "💡 To calculate costs for specific combinations:"
echo "   $0 --model aya-23-8B --target_lang ar"