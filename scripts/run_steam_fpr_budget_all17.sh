#!/bin/bash
#
# Extend the constant-budget-fraction null sweep (Table R3) from the 5 high-resource
# languages to all 17 target languages, running several languages concurrently.
#
# Why a wrapper: run_steam_fpr_budget.sh handles one language at a time and the full
# 17-language job is ~75 h serial (budget 38 at P=126 dominates). Per-text latency is
# dominated by CPU-bound GP refitting, so running one process per language is close to
# linear in the number of cores and brings the job to ~6-7 h wall-clock at 12-way.
#
# Every output file resumes independently (steam_bo_detector.run_null drops a partial
# trailing line and flushes per text), so this script is safe to interrupt and re-run.
#
# Usage:
#   ./scripts/run_steam_fpr_budget_all17.sh              # all 17 langs, 12-way
#   ./scripts/run_steam_fpr_budget_all17.sh 6            # all 17 langs, 6-way
#   ./scripts/run_steam_fpr_budget_all17.sh 6 "bn fa ta" # scoped, 6-way

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
LOG_DIR="$WORK_DIR/logs/fpr_budget_all17"

PARALLEL="${1:-12}"

# The 5 high-resource languages are already complete in gen/aya-23-8B-budgetfrac; they
# are listed anyway because run_steam_fpr_budget.sh skips any file already at 500 lines.
ALL_LANGS="fr de it es pt pl nl ru hi ko ja bn fa vi iw uk ta"
LANGS="${2:-$ALL_LANGS}"

mkdir -p "$LOG_DIR"

echo "Languages : $LANGS"
echo "Parallel  : $PARALLEL"
echo "Logs      : $LOG_DIR/{lang}.log"
echo ""

pids=()
fail_count=0

for LANG_CODE in $LANGS; do
    # Throttle to $PARALLEL concurrent language processes.
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 5
    done

    echo "--> launching $LANG_CODE"
    "$SCRIPT_DIR/run_steam_fpr_budget.sh" "$LANG_CODE" \
        > "$LOG_DIR/${LANG_CODE}.log" 2>&1 &
    pids+=("$!:$LANG_CODE")
done

for entry in "${pids[@]}"; do
    pid="${entry%%:*}"
    lang="${entry##*:}"
    if ! wait "$pid"; then
        echo "FAILED: $lang (see $LOG_DIR/${lang}.log)"
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "=== Completion check (expect 500 lines per file) ==="
for P in 33 126; do
    dir="$WORK_DIR/gen/aya-23-8B-budgetfrac/pool_${P}/kgw_seed0"
    n_done=0
    for LANG_CODE in $LANGS; do
        f="$dir/mc4.${LANG_CODE}.bo.hum.indep.z_score.jsonl"
        if [ -f "$f" ] && [ "$(wc -l < "$f" | tr -d ' ')" -ge 500 ]; then
            n_done=$((n_done + 1))
        else
            echo "  P=$P incomplete: $LANG_CODE"
        fi
    done
    echo "  P=$P complete: $n_done languages"
done

if [ "$fail_count" -gt 0 ]; then
    echo ""
    echo "$fail_count language(s) failed; re-run this script to resume."
    exit 1
fi

echo ""
echo "Done. Next:"
echo "  python3 analyze_fpr_calibration.py --model_abbr aya-23-8B-budgetfrac \\"
echo "    --pool_sizes 33 66 126 --pool_budgets 33:10 66:20 126:38 \\"
echo "    --no_tpr --require_langs 17 --emit_latex \\"
echo "    --out_dir results/fpr_calibration_budgetfrac"
