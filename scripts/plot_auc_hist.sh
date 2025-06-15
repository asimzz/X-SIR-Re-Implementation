#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/.."
FIGURE_DIR="$WORK_DIR/data/figures"
GEN_DIR="$WORK_DIR/gen"

MODEL_ABBR="llama-3.2-1B"
SEEDS=({0..49})
LANGS=("bn" "iw")

mkdir -p "$FIGURE_DIR"

echo "Plotting AUC histograms for $MODEL_ABBR, langs: ${LANGS[*]}"
python3 "$WORK_DIR/plot_auc_hist.py" \
    --model_abbr "$MODEL_ABBR" \
    --base_dir   "$GEN_DIR/$MODEL_ABBR" \
    --seeds      "${SEEDS[@]}" \
    --langs      "${LANGS[@]}" \
    --output     "$FIGURE_DIR/${MODEL_ABBR}_auc_hist.png"
