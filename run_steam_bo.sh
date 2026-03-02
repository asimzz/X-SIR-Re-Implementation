#!/bin/bash
# STEAM BO Detection Pipeline (KGW)
set -e

# Configuration
BASE_MODEL="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
N_INITIAL=3
MAX_EVALUATIONS=15
NUM_TEXTS=500

TARGET_LANGS=("fr" "de" "es" "it" "pt" "nl" "pl" "ru" "hi" "ko" "ja" "bn" "fa" "vi" "iw" "uk" "ta")

# Directories
INPUT_DIR="gen/${MODEL_ABBR}/kgw_seed0"
OUTPUT_DIR="gen/${MODEL_ABBR}/kgw_seed0"

# Allow running a single language
if [[ -n "$1" ]]; then
    TARGET_LANGS=("$1")
fi

for TGT_LANG in "${TARGET_LANGS[@]}"; do
    echo "=== Processing: $TGT_LANG ==="

    python3 steam_bo_detector.py \
        --base_model "$BASE_MODEL" \
        --tgt_lang "$TGT_LANG" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --n_initial "$N_INITIAL" \
        --max_evaluations "$MAX_EVALUATIONS" \
        --num_texts "$NUM_TEXTS"

    echo ""
done
