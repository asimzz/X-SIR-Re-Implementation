#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data

# Parameters for X-KGW
MAPPING_DIR=$DATA_DIR/mapping/xkgw
mkdir -p "$MAPPING_DIR"

MODEL_NAMES=(
"google/gemma-4-E4B"
)

MODEL_ABBRS=(
    "gemma4-E4B"
)

SEEDS=(0 42 123)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        echo "Generating X-KGW semantic mappings for $MODEL_NAME with seed $SEED"

        python3 $WORK_DIR/src_watermark/xkgw/generate_mappings.py \
            --model "$MODEL_NAME" \
            --dictionary "$DATA_DIR/dictionary/dictionary.txt" \
            --output_file "$MAPPING_DIR/xkgw_mapping_${MODEL_ABBR}_seed${SEED}.json" \
            --seed "$SEED"
    done
done
