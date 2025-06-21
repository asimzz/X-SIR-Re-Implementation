#!/bin/bash

set -e
set -u

# ─── CONFIGURABLE PARAMETERS ───────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
REF_SEEDS=({0..19})                 # The reference seeds to use
TGT_LANG="bn"                         # Target language (e.g., bn for Bengali)
MODEL_ABBR="llama-3.2-1B"             # Abbreviation used in your `gen` folder
BASE_DIR="./gen"                      # Root path to your gen directory
FIGURE_DIR=$WORK_DIR/data/figures         # Directory to save the output plot



# ─── JOIN SEED LIST INTO ARG STRING ────────────────────────────────────────
SEED_ARGS=""
for seed in "${REF_SEEDS[@]}"; do
  SEED_ARGS="$SEED_ARGS $seed"
done

# ─── CONSTRUCT OUTPUT PATH ─────────────────────────────────────────────────
OUTPUT_PATH="${FIGURE_DIR}/heatmap_${MODEL_ABBR}_${TGT_LANG}.png"

# ─── RUN PYTHON SCRIPT ─────────────────────────────────────────────────────
echo "📊 Generating AUC heatmap for model=${MODEL_ABBR}, lang=${TGT_LANG}..."
python3 "$WORK_DIR/src_watermark/xsir/plot_patched_heatmap.py" \
  --model_abbr "$MODEL_ABBR" \
  --base_dir "$BASE_DIR" \
  --seeds $SEED_ARGS \
  --tgt_lang "$TGT_LANG" \
  --output "$OUTPUT_PATH"

echo "✅ Done. Heatmap saved to $OUTPUT_PATH"
