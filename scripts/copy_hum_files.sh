#!/bin/bash

# Source folder
SRC="gen/aya-23-8B/sir"
# Files to copy

TGT_LANGS=(
    "de"
    "fr"
    "zh"
    "ja"
)

# Loop over all seed_* folders except seed_0
for DIR in gen/aya-23-8B/xsir/*; do
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