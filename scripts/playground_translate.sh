#!/bin/bash
set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
FIGURE_DIR=$WORK_DIR/data/figures
GEN_DIR=$WORK_DIR/gen

MODELS=(
    "llama-3.2-1B"
    # "aya-23-8B"
    # "bloom-7b1"
    # "xglm-564M"
)

mkdir -p $FIGURE_DIR

ORG_LANGS=(
    "en" # English
    # High-resource languages
    "fr" # French
    "de" # German
    "it" # Italian
    "es" # Spanish
    "pt" # Portuguese
    # Medium-resource languages
    "pl" # Polish
    "nl" # Dutch
    "ru" # Russian
    "hi" # Hindi
    "ko" # Korean
    "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    "vi" # Vietnamese
    "iw" # Hebrew
    "uk" # Ukrainian
    "ta" # Tamil
)


for MODEL_ABBR in "${MODELS[@]}"; do
  for ORG_LANG in "${ORG_LANGS[@]}"; do
    echo "Generating plots for ${MODEL_ABBR} with original language ${ORG_LANG}..."
    OUTPUT_PNG="${FIGURE_DIR}/translation_attack_${MODEL_ABBR}_kgw_${ORG_LANG}.png"
    python3 $WORK_DIR/playground_translate_3.py \
      --model_abbr "${MODEL_ABBR}" \
      --base_dir "${GEN_DIR}" \
      --output "${OUTPUT_PNG}" \
      --org_lang "${ORG_LANG}"
  done
done