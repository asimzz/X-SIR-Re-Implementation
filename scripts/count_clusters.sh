#!/bin/bash

# Exit on any error or unset variable
set -e
set -u

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
MAPPING_DIR="$DATA_DIR/mapping/xsir"

# Model configuration
TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE=32

# Model names and abbreviations
MODEL_NAMES=(
    "CohereForAI/aya-23-8B"
    # "meta-llama/Llama-3.2-1B"
    # "LLaMAX/LLaMAX3-8B"
)
MODEL_ABBRS=(
    "aya-23-8B"
    # "llama-3.2-1B"
    # "llamax3-8B"
)



# Settings
WATERMARK_METHODS=("sir")
ITERS=({0..20})
TGT_LANGS=(
    "de"
    "fr"
    "zh"
    "ja"
)

# Validate model list lengths
if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "❌ MODEL_NAMES and MODEL_ABBRS length mismatch."
    exit 1
fi

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for ITER in "${ITERS[@]}"; do
        echo "🔍 Counting clusters for iteration $ITER"
        python3 $WORK_DIR/count_clusters.py \
            --clusters_file $MAPPING_DIR/$ITER/300_mapping_${MODEL_ABBR}_clusters.json
    done
done