#!/bin/bash
set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."

# Model configuration
BASE_MODEL="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
WATERMARK="kgw_seed0"

INPUT_DIR="$WORK_DIR/gen/$MODEL_ABBR/$WATERMARK"
OUTPUT_DIR="$INPUT_DIR"
GAMMA_LANG_FILE="$INPUT_DIR/gamma_lang.json"

# BO settings
N_INITIAL=3
MAX_EVALUATIONS=15
NUM_TEXTS=500

# 17 target languages (first step of attack)
TGT_LANGS=(
    "fr" "de" "it" "es" "pt"
    "pl" "nl" "ru" "hi" "ko" "ja"
    "bn" "fa" "vi" "iw" "uk" "ta"
)

# 3 pivot languages (second step of attack)
PVT_LANGS=("de" "ko" "bn")

for TGT_LANG in "${TGT_LANGS[@]}"; do
    for PVT_LANG in "${PVT_LANGS[@]}"; do
        # Skip if pivot == target
        if [ "$PVT_LANG" == "$TGT_LANG" ]; then
            continue
        fi

        PREFIX="${TGT_LANG}-${PVT_LANG}-pivot"
        MOD_FILE="$INPUT_DIR/mc4.${PREFIX}.mod.jsonl"
        HUM_FILE="$INPUT_DIR/mc4.${PREFIX}.hum.jsonl"

        # Skip if input doesn't exist
        if [ ! -f "$MOD_FILE" ]; then
            echo "Skipping $PREFIX: input not found"
            continue
        fi

        echo "=== STEAM-BO: $TGT_LANG -> $PVT_LANG ==="
        python3 "$WORK_DIR/steam_bo_detector.py" \
            --base_model "$BASE_MODEL" \
            --tgt_lang "$TGT_LANG" \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --gamma_lang_file "$GAMMA_LANG_FILE" \
            --input_mod "$MOD_FILE" \
            --input_hum "$HUM_FILE" \
            --output_prefix "$PREFIX" \
            --n_initial "$N_INITIAL" \
            --max_evaluations "$MAX_EVALUATIONS" \
            --num_texts "$NUM_TEXTS"
        echo ""
    done
done
