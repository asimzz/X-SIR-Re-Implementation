#!/bin/bash

# Source folder
SRC="gen/llama-3.2-1B/kgw_seed0"
# Files to copy

TGT_LANGS=(
    # High-resource languages
    "fr" # French
    "de" # German
    "it" # Italian
    "es" # Spanish
    "pt" # Portuguese
    # Medium-resource languages
    "pl" # Polish
    "nl" # Dutch
    "ru" # Russian
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

# Loop over all seed_* folders except seed_0
for DIR in gen/aya-23-8B/xsir_seed*; do
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