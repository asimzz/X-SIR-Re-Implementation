#!/bin/bash
#
# Reset stale STEAM-BO outputs for the 17 target languages.
#
# The base kgw_seed0 positives and the pool_33/pool_66 positives, plus any
# already-generated independent nulls, were produced with the BUGGY 84-pivot pool
# (before language_code_converter.py was completed). With the fixed converter the
# full pool is 125 pivots (matching language_clusters / the paper's P=126), so those
# outputs must be regenerated. This script ARCHIVES them (move, not delete — git
# history also retains them) so the resumable runners regenerate them cleanly.
#
# It touches ONLY the 17 target languages; other languages in kgw_seed0 are left
# alone. pool_126/kgw_seed0 is a symlink to kgw_seed0, so clearing the base also
# clears the P=126 view.
#
# After running this:
#   ./scripts/run_steam_pool_sweep.sh     # regenerate positives for pools 33/66/126
#   ./scripts/run_steam_fpr_null.sh       # regenerate independent nulls for 33/66/126
#   python3 analyze_fpr_calibration.py --model_abbr aya-23-8B --methods kgw --seeds 0

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
GEN="$WORK_DIR/gen/aya-23-8B"
BACKUP="$WORK_DIR/gen/_stale_pool84_backup"

TGT_LANGS=(fr de it es pt pl nl ru hi ko ja bn fa vi iw uk ta)

# Subdirs holding pool-dependent BO outputs. pool_126 is a symlink to kgw_seed0,
# so it is intentionally NOT listed separately (clearing kgw_seed0 covers it).
SUBDIRS=(kgw_seed0 pool_33/kgw_seed0 pool_66/kgw_seed0)

# The stale, pool-dependent output patterns (per target language).
PATTERNS=(
    "mc4.%s.bo.z_score.jsonl"            # watermarked positives (max-over-search)
    "mc4.%s.bo.hum.z_score.jsonl"        # old paired-pivot nulls (unused, but stale)
    "mc4.%s.bo.hum.indep.z_score.jsonl"  # independent nulls (if already started)
)

moved=0
for sub in "${SUBDIRS[@]}"; do
    d="$GEN/$sub"
    [ -d "$d" ] || continue
    for lang in "${TGT_LANGS[@]}"; do
        for pat in "${PATTERNS[@]}"; do
            f="$d/$(printf "$pat" "$lang")"
            if [ -f "$f" ]; then
                dest="$BACKUP/$sub"
                mkdir -p "$dest"
                mv "$f" "$dest/"
                moved=$((moved + 1))
            fi
        done
    done
done

echo "Archived $moved stale (84-pivot) output file(s) → $BACKUP"
echo "Next: ./scripts/run_steam_pool_sweep.sh  then  ./scripts/run_steam_fpr_null.sh"
