#!/bin/bash

# Number of total reference seeds and seeds per batch
TOTAL_REF_SEEDS=20
BATCH_SIZE=10
NUM_BATCHES=$((TOTAL_REF_SEEDS / BATCH_SIZE))

for ((batch=0; batch<NUM_BATCHES; batch++)); do
  START_SEED=$((batch * BATCH_SIZE))
  END_SEED=$((START_SEED + BATCH_SIZE - 1))
  SESSION_NAME="xsir_batch${batch}"

  echo "🚀 Launching tmux session: $SESSION_NAME to handle ref_seeds ${START_SEED}-${END_SEED}"

  # Start tmux session and activate venv before running batch
  tmux new-session -d -s "$SESSION_NAME" "source .venv/bin/activate && bash scripts/run_batch.sh $START_SEED $END_SEED"

  sleep 2
done

echo "✅ All tmux sessions launched. Use 'tmux ls' to monitor."
