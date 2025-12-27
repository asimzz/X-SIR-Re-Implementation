#!/bin/bash
set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
RESULTS_DIR=$WORK_DIR/hierarchical_optimization_results

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

echo "Starting HIERARCHICAL STEAM Pivot Language Optimization for ${#TGT_LANGS[@]} target languages"
echo "🧬 This approach selects optimal language representatives from each cluster"
echo "📊 Results will be saved to: $RESULTS_DIR"
echo "======================================="

for TGT_LANG in "${TGT_LANGS[@]}"; do
    echo ""
    echo "🎯 Hierarchical optimization for target: $TGT_LANG"
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

    RESULT_FILE="$RESULTS_DIR/hierarchical_optimization_${TGT_LANG}.json"

    echo "Target language code: $TGT_LANG -> $URIEL_LANG"
    echo "Output file: $RESULT_FILE"

    # Run hierarchical optimization with intelligent cluster utilization
    python3 "$WORK_DIR/optimize_hierarchical_pivots.py" \
        --target_lang "$URIEL_LANG" \
        --output_file "$RESULT_FILE" \
        --n_calls 20 \
        --max_clusters 2 \
        --langs_per_cluster 2

    if [ $? -eq 0 ]; then
        echo "✅ Hierarchical optimization completed for $TGT_LANG"
        echo "📊 Results saved to: $RESULT_FILE"
    else
        echo "❌ Hierarchical optimization failed for $TGT_LANG"
    fi

    echo "---------------------------------------"
done

echo ""
echo "🎉 HIERARCHICAL STEAM Pivot Optimization Complete!"
echo "📁 All results saved in: $RESULTS_DIR"
echo "📋 Summary:"

# Generate summary
for TGT_LANG in "${TGT_LANGS[@]}"; do
    RESULT_FILE="$RESULTS_DIR/hierarchical_optimization_${TGT_LANG}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "  ✅ $TGT_LANG: $(basename $RESULT_FILE)"
    else
        echo "  ❌ $TGT_LANG: FAILED"
    fi
done

echo ""
echo "🔬 Key Differences from Simple Approach:"
echo "1. ✨ Selects OPTIMAL language representatives from each cluster"
echo "2. ⚡ More efficient - uses 2-4 languages instead of entire clusters"
echo "3. 🎯 Better genetic distance optimization within cluster boundaries"
echo "4. 📈 Should provide better STEAM detection performance"
echo ""
echo "📊 Next steps:"
echo "1. Compare these hierarchical results with simple cluster results"
echo "2. Test both approaches in translation experiments"
echo "3. Determine which gives better STEAM detection performance"