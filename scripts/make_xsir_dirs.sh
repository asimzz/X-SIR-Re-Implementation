#!/bin/bash
# Create empty folders for each target language and xsir_seed folders in each model folder under gen

GEN_DIR="$(dirname "$0")/../gen"
TGT_LANGS=("en" "fr" "de" "zh" "ja")
SEEDS=(0 42 123)

for MODEL_DIR in "$GEN_DIR"/*; do
    [ -d "$MODEL_DIR" ] || continue
    for LANG in "${TGT_LANGS[@]}"; do
        LANG_DIR="$MODEL_DIR/$LANG"
        mkdir -p "$LANG_DIR"
        for SEED in "${SEEDS[@]}"; do
            mkdir -p "$LANG_DIR/xsir_seed$SEED"
        done
    done
done
