#!/bin/bash
# Compute γ_lang (language-specific green token fraction) for all models
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."

MODEL_NAMES=("google/gemma-4-E4B")
MODEL_ABBRS=("gemma4-E4B")
SEEDS=(0)

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        OUTPUT_FILE="$WORK_DIR/gen/${MODEL_ABBR}/kgw_seed${SEED}/gamma_lang.json"

        echo "=== Computing γ_lang for ${MODEL_ABBR} (seed=${SEED}) ==="

        python3 "$WORK_DIR/compute_gamma_lang.py" \
            --base_model "$MODEL_NAME" \
            --input_dir "$WORK_DIR/data/dataset/mc4" \
            --output_file "$OUTPUT_FILE" \
            --gamma 0.25 \
            --seed "$SEED" \
            --seeding_scheme minhash

        echo "Saved to $OUTPUT_FILE"
        echo ""
    done
done
