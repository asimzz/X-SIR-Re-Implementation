#!/bin/bash

# Exit on any error or unset variable
set -e
set -u

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"
MAPPING_DIR="$DATA_DIR/mapping/xsir"
FIGURE_DIR="$WORK_DIR/data/figures"

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

    echo "🔍 Plotting AUC vs Cluster Size for $MODEL_NAME"
    python3 $WORK_DIR/plot_auc_vs_clusters_size.py \
        --base_dir "$GEN_DIR/$MODEL_ABBR" \
        --model_abbr "$MODEL_ABBR" \
        --mapping_dir "$MAPPING_DIR" \
        --output_dir "$FIGURE_DIR"
done