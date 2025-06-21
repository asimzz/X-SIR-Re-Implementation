#!/bin/bash

START_SEED=$1
END_SEED=$2

echo "🎯 Running seeds from $START_SEED to $END_SEED"

for ((ref_seed=START_SEED; ref_seed<=END_SEED; ref_seed++)); do
  echo "🔁 Running ref_seed $ref_seed"
  bash scripts/run_all.sh "$ref_seed"
done
