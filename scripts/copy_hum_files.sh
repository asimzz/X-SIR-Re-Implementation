#!/bin/bash

# Source folder
SRC="gen/llama-3.2-1B/xsir/seed_0"
# Files to copy
FILES=("mc4.en-bn.hum.jsonl" "mc4.en-iw.hum.jsonl")

# Loop over all seed_* folders except seed_0
for DIR in gen/llama-3.2-1B/xsir/seed_*; do
    # Skip seed_0
    if [[ "$DIR" == "$SRC" ]]; then
        continue
    fi
    for FILE in "${FILES[@]}"; do
        # Only copy if file does not exist
        if [[ ! -f "$DIR/$FILE" ]]; then
            cp "$SRC/$FILE" "$DIR/"
            echo "Copied $FILE to $DIR"
        else
            echo "Skipped $DIR/$FILE (already exists)"
        fi
    done
done