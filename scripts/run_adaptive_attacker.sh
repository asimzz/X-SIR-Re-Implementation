#!/bin/bash
# Adaptive oracle-attacker analysis (reviewer rebuttal).
# Reuses already-computed STEAM-BO outputs across the 47 attack pivots
# (17 NEW_SUPPORTED_LANGS + 30 new langs from generate_with_watermark_translate.sh).

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

MODELS=("aya-23-8B")
METHODS=("kgw")
SEEDS=(0)
NUM_TEXTS=500

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo ">>> $model / $method / seed $seed (n=$NUM_TEXTS)"
      python3 analyze_adaptive_attacker.py \
        --model_abbr "$model" \
        --method "$method" \
        --seed "$seed" \
        --num_texts "$NUM_TEXTS" \
        --out_dir results/adaptive_attacker
    done
  done
done
