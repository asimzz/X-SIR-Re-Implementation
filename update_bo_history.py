#!/usr/bin/env python3
import json
import sys
from select_next_language import compute_mean_zscore

def update_bo_history(target_lang, selected_lang, out_dir, history_file='bo_history.json'):
    """Update BO history with actual z-score after evaluation."""

    # Load current history
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Find the last entry (should be the selected language with placeholder z-score)
    if history['languages'][-1] == selected_lang:
        # Compute actual z-score
        mod_file = f"{out_dir}/mc4.{target_lang}-{selected_lang}-back.mod.z_score.jsonl"
        val_file = f"{out_dir}/mc4.{target_lang}-{selected_lang}-back.val.z_score.jsonl"

        try:
            actual_zscore = compute_mean_zscore(mod_file, val_file)
            history['zscores'][-1] = actual_zscore

            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f)

            print(f"Updated z-score for {selected_lang}: {actual_zscore:.4f}")
        except Exception as e:
            print(f"Failed to compute z-score for {selected_lang}: {e}")
    else:
        print(f"WARNING: Expected {selected_lang} in history, found {history['languages'][-1]}")

if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print("Usage: python update_bo_history.py <target_lang> <selected_lang> <out_dir> [history_file]")
        sys.exit(1)

    target_lang, selected_lang, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    history_file = sys.argv[4] if len(sys.argv) == 5 else 'bo_history.json'
    update_bo_history(target_lang, selected_lang, out_dir, history_file)