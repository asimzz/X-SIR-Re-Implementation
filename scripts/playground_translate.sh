#!/bin/bash
set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
FIGURE_DIR=$WORK_DIR/data/figures
GEN_DIR=$WORK_DIR/gen

MODELS=(
    "llama-3.2-1B"
    # "bloom-7b1"
    "aya-23-8B"
)

mkdir -p $FIGURE_DIR

for MODEL_ABBR in "${MODELS[@]}"; do
  OUTPUT_PNG="${FIGURE_DIR}/translation_attack_${MODEL_ABBR}_google_translate.png"
  echo "Generating plot for ${MODEL_ABBR}..."
  python3 $WORK_DIR/playground_translate.py \
    --model_abbr "${MODEL_ABBR}" \
    --base_dir "${GEN_DIR}" \
    --output "${OUTPUT_PNG}"
done
