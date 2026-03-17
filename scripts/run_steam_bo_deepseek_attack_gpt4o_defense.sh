#!/bin/bash
# Attacker-Defender Mismatch: Attack=DeepSeek, Defense=GPT-4o-mini
set -e

BASE_MODEL="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
N_INITIAL=3
MAX_EVALUATIONS=20
NUM_TEXTS=500

TARGET_LANGS=("de" "hi" "iw")

GEN_DIR="gen/${MODEL_ABBR}/kgw_seed0"
GAMMA_LANG_FILE="${GEN_DIR}/gamma_lang.json"

if [[ -n "$1" ]]; then
    TARGET_LANGS=("$1")
fi

for TGT_LANG in "${TARGET_LANGS[@]}"; do
    echo "=== DeepSeek attack / GPT-4o-mini defense: $TGT_LANG ==="

    python3 steam_bo_detector.py \
        --base_model "$BASE_MODEL" \
        --tgt_lang "$TGT_LANG" \
        --input_dir "$GEN_DIR" \
        --output_dir "$GEN_DIR" \
        --gamma_lang_file "$GAMMA_LANG_FILE" \
        --n_initial "$N_INITIAL" \
        --max_evaluations "$MAX_EVALUATIONS" \
        --num_texts "$NUM_TEXTS" \
        --translator gpt4o \
        --input_mod "${GEN_DIR}/mc4.en-${TGT_LANG}.deepseek.jsonl" \
        --input_hum "${GEN_DIR}/mc4.en-${TGT_LANG}.hum.jsonl" \
        --output_prefix "deepseek.gpt4o.${TGT_LANG}"

    echo ""
done
