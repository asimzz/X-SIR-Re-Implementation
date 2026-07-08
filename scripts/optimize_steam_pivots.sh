#!/bin/bash
set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
RESULTS_DIR=$WORK_DIR/steam_optimization_results

# Create results directory
mkdir -p "$RESULTS_DIR"

# Target languages from eval_back_translate.sh
TGT_LANGS=(
    # High-resource languages
    "fr"
    "de"
    "it"
    "es"
    "pt"
    # Medium-resource languages
    "pl"
    "nl"
    "ru"
    "hi"
    "ko"
    "ja"
    # Low-resource languages
    "bn"
    "fa"
    "vi"
    "iw"  # Hebrew
    "uk"
    "ta"
)

echo "Starting STEAM Pivot Language Optimization for ${#TGT_LANGS[@]} target languages"
echo "Results will be saved to: $RESULTS_DIR"
echo "======================================="

for TGT_LANG in "${TGT_LANGS[@]}"; do
    echo ""
    echo "🎯 Optimizing pivot languages for target: $TGT_LANG"
    echo "---------------------------------------"

    # Convert 2-letter to 3-letter language codes for URIEL
    case "$TGT_LANG" in
        "fr") URIEL_LANG="fra" ;;
        "de") URIEL_LANG="deu" ;;
        "it") URIEL_LANG="ita" ;;
        "es") URIEL_LANG="spa" ;;
        "pt") URIEL_LANG="por" ;;
        "pl") URIEL_LANG="pol" ;;
        "nl") URIEL_LANG="nld" ;;
        "ru") URIEL_LANG="rus" ;;
        "hi") URIEL_LANG="hin" ;;
        "ko") URIEL_LANG="kor" ;;
        "ja") URIEL_LANG="jpn" ;;
        "bn") URIEL_LANG="ben" ;;
        "fa") URIEL_LANG="fas" ;;
        "vi") URIEL_LANG="vie" ;;
        "iw") URIEL_LANG="heb" ;;
        "uk") URIEL_LANG="ukr" ;;
        "ta") URIEL_LANG="tam" ;;
        *) URIEL_LANG="$TGT_LANG" ;;
    esac

    RESULT_FILE="$RESULTS_DIR/steam_optimization_${TGT_LANG}.json"

    echo "Target language code: $TGT_LANG -> $URIEL_LANG"
    echo "Output file: $RESULT_FILE"

    # Run genetic distance optimization with all 4 strategies
    python3 "$WORK_DIR/optimize_steam_pivots.py" \
        --target_lang "$URIEL_LANG" \
        --output_file "$RESULT_FILE" \
        --n_calls 20 \
        --max_clusters 2

    if [ $? -eq 0 ]; then
        echo "✅ Optimization completed for $TGT_LANG"
        echo "📊 Results saved to: $RESULT_FILE"
    else
        echo "❌ Optimization failed for $TGT_LANG"
    fi

    echo "---------------------------------------"
done

echo ""
echo "🎉 STEAM Pivot Optimization Complete!"
echo "📁 All results saved in: $RESULTS_DIR"
echo "📋 Summary:"

# Generate summary
for TGT_LANG in "${TGT_LANGS[@]}"; do
    RESULT_FILE="$RESULTS_DIR/steam_optimization_${TGT_LANG}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "  ✅ $TGT_LANG: $(basename $RESULT_FILE)"
    else
        echo "  ❌ $TGT_LANG: FAILED"
    fi
done

echo ""
echo "🔬 Next steps:"
echo "1. Use optimized pivot languages instead of fixed sets"
echo "2. Apply STEAM method with genetic distance-optimized pivots"
echo "3. Compare performance against old fixed pivot approach"