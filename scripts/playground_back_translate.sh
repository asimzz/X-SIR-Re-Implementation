#!/bin/bash
set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
FIGURE_DIR=$WORK_DIR/data/figures
GEN_DIR=$WORK_DIR/gen

MODELS=(
    "llama-3.2-1B"
    "aya-23-8B"
    # "bloom-7b1"
    # "xglm-564M"
)

mkdir -p $FIGURE_DIR

for MODEL_ABBR in "${MODELS[@]}"; do
    echo "Generating back-translation figure for $MODEL_ABBR"
    OUTPUT_PNG="${FIGURE_DIR}/back_translation_defense_${MODEL_ABBR}_kgw_without_ta.pdf"
    python3 $WORK_DIR/playground_back_translate.py \
      --model_abbr "${MODEL_ABBR}" \
      --base_dir "${GEN_DIR}" \
      --output "${OUTPUT_PNG}"
done