#!/bin/bash
#
# Evaluation companion to run_steam_pool_sweep.sh.
# Reads pool-sweep outputs and prints AUC / TPR@FPR / F1 metrics, tagged with
# the pool size for downstream parsing by analyze_pool_size.py.
#
# Usage: ./scripts/evaluate_detection_pool_sweep.sh

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
GEN_DIR="$WORK_DIR/gen"

MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)
MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

WATERMARK_METHODS=("kgw")
SEEDS=(0)

TGT_LANGS=(
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

POOL_SIZES=(33 66 133)

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for POOL_SIZE in "${POOL_SIZES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
                POOL_DIR="$GEN_DIR/$MODEL_ABBR/pool_${POOL_SIZE}/${WATERMARK_METHOD}_seed${SEED}"

                for TGT_LANG in "${TGT_LANGS[@]}"; do
                    WM_FILE="$POOL_DIR/mc4.${TGT_LANG}.bo.z_score.jsonl"
                    HM_FILE="$POOL_DIR/mc4.${TGT_LANG}.bo.hum.z_score.jsonl"

                    if [ ! -f "$WM_FILE" ] || [ ! -f "$HM_FILE" ]; then
                        continue
                    fi

                    echo "pool_${POOL_SIZE} $MODEL_NAME $WATERMARK_METHOD (seed=$SEED) STEAM-BO ($TGT_LANG)"
                    python3 "$WORK_DIR/eval_detection.py" \
                        --hm_zscore "$HM_FILE" \
                        --wm_zscore "$WM_FILE"
                done
                echo "======================================="
            done
        done
    done
done
