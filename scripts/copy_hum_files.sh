#!/bin/bash

# Source folder
SRC="gen/llama-3.2-1B/xsir_seed0"
# Files to copy

TGT_LANGS=("it" "es" "pt" "pl" "nl" "hr" "cs" "da" "ko" "ar")

# Loop over all seed_* folders except seed_0
for DIR in gen/llamax3-8B/xkgw_seed0; do
    # Skip seed_0
    if [[ "$DIR" == "$SRC" ]]; then
        continue
    fi
    for TGT_LANG in "${TGT_LANGS[@]}"; do
        # Only copy if file does not exist
        FILE="mc4.en-${TGT_LANG}.hum.jsonl"
        if [[ ! -f "$DIR/$FILE" ]]; then
            cp "$SRC/$FILE" "$DIR/"
            echo "Copied $FILE to $DIR"
        else
            echo "Skipped $DIR/$FILE (already exists)"
        fi
    done
done