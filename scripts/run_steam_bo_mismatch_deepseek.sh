#!/bin/bash
# STEAM BO Mismatch Experiment: Attack=GPT-4o mini, Defense=DeepSeek
#
# Attack: GPT-4o mini translates watermarked English text to {de, hi, iw}
# Defense: STEAM-BO back-translates using DeepSeek to detect watermark
#
# Requires: DEEP_SEEK_API_KEY environment variable
#
# Input:  mc4.en-{tgt}.gpt4o.jsonl         (GPT-4o attacked watermarked text)
#         mc4.en-{tgt}.hum.z_score.jsonl    (human text in target lang, used as hum input)
# Output: mc4.en-{tgt}.gpt4o.deepseek.bo.z_score.jsonl
#         mc4.en-{tgt}.gpt4o.deepseek.bo.hum.z_score.jsonl
set -e

# Configuration
BASE_MODEL="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
N_INITIAL=3
MAX_EVALUATIONS=20
NUM_TEXTS=500

TARGET_LANGS=("de" "es" "hi" "ru" "iw" "bn")

# Directories
GEN_DIR="gen/${MODEL_ABBR}/kgw_seed0"
GAMMA_LANG_FILE="${GEN_DIR}/gamma_lang.json"

# Allow running a single language
if [[ -n "$1" ]]; then
    TARGET_LANGS=("$1")
fi

for TGT_LANG in "${TARGET_LANGS[@]}"; do
    echo "=== Mismatch (DeepSeek defense): $TGT_LANG ==="

    python3 steam_bo_detector.py \
        --base_model "$BASE_MODEL" \
        --tgt_lang "$TGT_LANG" \
        --input_dir "$GEN_DIR" \
        --output_dir "$GEN_DIR" \
        --gamma_lang_file "$GAMMA_LANG_FILE" \
        --n_initial "$N_INITIAL" \
        --max_evaluations "$MAX_EVALUATIONS" \
        --num_texts "$NUM_TEXTS" \
        --translator deepseek \
        --input_mod "${GEN_DIR}/mc4.en-${TGT_LANG}.gpt4o.jsonl" \
        --input_hum "${GEN_DIR}/mc4.en-${TGT_LANG}.hum.z_score.jsonl" \
        --output_prefix "en-${TGT_LANG}.gpt4o.deepseek"

    echo ""
done
