#!/bin/bash
# STEAM BO Detection Pipeline (KGW)
#
# Prerequisites:
#   1. Run compute_gamma_lang.py to generate gamma_lang.json:
#      python3 compute_gamma_lang.py \
#        --base_model CohereForAI/aya-23-8B \
#        --input_dir data/dataset/mc4 \
#        --output_file gen/aya-23-8B/kgw_seed0/gamma_lang.json
set -e

# Configuration
BASE_MODEL="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
N_INITIAL=3
MAX_EVALUATIONS=15
NUM_TEXTS=500

# Gemini backtranslation settings
TRANSLATOR="gemini"
GEMINI_MODEL="gemini-2.5-flash"
GEMINI_TEMPERATURE=0.2

if [[ "$TRANSLATOR" == "gemini" && -z "$GEMINI_API_KEY" ]]; then
    echo "ERROR: GEMINI_API_KEY is not set. Export it before running with the gemini translator." >&2
    exit 1
fi

TARGET_LANGS=(
   "de"
)

# Directories
INPUT_DIR="gen/${MODEL_ABBR}/kgw_seed0"
OUTPUT_DIR="gen/${MODEL_ABBR}/kgw_seed0/gemini_bo"
GAMMA_LANG_FILE="gen/${MODEL_ABBR}/kgw_seed0/gamma_lang.json"

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
        --gamma_lang_file "$GAMMA_LANG_FILE" \
        --n_initial "$N_INITIAL" \
        --max_evaluations "$MAX_EVALUATIONS" \
        --num_texts "$NUM_TEXTS" \
        --translator "$TRANSLATOR" \
        --gemini_model "$GEMINI_MODEL" \
        --gemini_temperature "$GEMINI_TEMPERATURE"

    echo ""
done
