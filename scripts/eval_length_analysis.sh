#!/bin/bash

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
GEN_DIR=$WORK_DIR/gen

MODEL_NAMES=(
    "CohereForAI/aya-23-8B"
)

MODEL_ABBRS=(
    "aya-23-8B"
)

WATERMARK_METHODS=("kgw")
TGT_LANGS=(
    # High-resource languages
    "de" # German
    "es" # Spanish
    # Medium-resource languages
    "ru" # Russian
    "hi" # Hindi
    "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    "iw" # Hebrew
    )
SEEDS=(0)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

# Create results directory if it doesn't exist
RESULTS_DIR=$WORK_DIR/results_length_analysis
mkdir -p $RESULTS_DIR

echo "======================================="
echo "📊 WATERMARK STRENGTH BY TEXT LENGTH"
echo "======================================="
echo "This analysis evaluates how text length affects"
echo "watermark detection performance across languages."
echo ""
echo "Length Categories (Dynamic per language):"
echo "- Short: Bottom 1/3 of texts by token length"
echo "- Medium: Middle 1/3 of texts by token length"
echo "- Long: Top 1/3 of texts by token length"
echo ""
echo "Uses percentile-based binning to ensure equal"
echo "sample sizes and reliable AUC scores."
echo ""

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do

            WATERMARK_DIR=$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}

            echo "======================================="
            echo "🔍 Model: $MODEL_NAME"
            echo "🔧 Watermark Method: $WATERMARK_METHOD (seed=$SEED)"
            echo "======================================="

            # Create a summary file for this model/method combination
            SUMMARY_FILE="$RESULTS_DIR/${MODEL_ABBR}_${WATERMARK_METHOD}_seed${SEED}_length_analysis_summary.txt"
            echo "Length Analysis Summary: $MODEL_NAME $WATERMARK_METHOD (seed=$SEED)" > $SUMMARY_FILE
            echo "Generated on: $(date)" >> $SUMMARY_FILE
            echo "=======================================" >> $SUMMARY_FILE
            echo "" >> $SUMMARY_FILE

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo ""
                echo ">>> Target Language: $TGT_LANG <<<"
                echo ""

                # Check if the required files exist
                HUM_FILE="$WATERMARK_DIR/mc4.en-${TGT_LANG}.hum.z_score.jsonl"
                WM_FILE="$WATERMARK_DIR/mc4.en-${TGT_LANG}.mod.z_score.jsonl"

                if [ ! -f "$HUM_FILE" ] || [ ! -f "$WM_FILE" ]; then
                    echo "Warning: Required files not found for $TGT_LANG, skipping..."
                    echo "Expected files:"
                    echo "  - $HUM_FILE"
                    echo "  - $WM_FILE"
                    continue
                fi

                echo "Analyzing watermark strength by text length for $TGT_LANG..."

                # Set output paths
                PLOT_FILE="$RESULTS_DIR/${MODEL_ABBR}_${WATERMARK_METHOD}_seed${SEED}_${TGT_LANG}_length_analysis.png"
                CSV_FILE="$RESULTS_DIR/${MODEL_ABBR}_${WATERMARK_METHOD}_seed${SEED}_${TGT_LANG}_length_analysis.csv"

                # Run length analysis evaluation
                python3 $WORK_DIR/evaluate_length_analysis.py \
                    --tgt_lang "$TGT_LANG" \
                    --base_wm_dir "$WATERMARK_DIR" \
                    --output_plot "$PLOT_FILE" \
                    --output_csv "$CSV_FILE" | tee -a $SUMMARY_FILE

                echo "" >> $SUMMARY_FILE
                echo "=======================================" >> $SUMMARY_FILE
                echo "" >> $SUMMARY_FILE

            done

            echo ""
            echo "✅ Completed analysis for $MODEL_NAME $WATERMARK_METHOD (seed=$SEED)"
            echo "📄 Summary saved to: $SUMMARY_FILE"
            echo ""
        done
    done
done

echo ""
echo "🎉 All length analysis evaluations completed!"
echo "📊 Results saved to: $RESULTS_DIR"
echo ""

echo "📊 Aggregating results into comprehensive tables..."
python3 $WORK_DIR/aggregate_length_tables.py \
    --results_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/aggregated_length_results.csv"

echo ""
echo "📁 Generated files:"
find $RESULTS_DIR -name "*.png" -o -name "*.txt" -o -name "*.csv" | sort
echo ""
echo "📊 Individual CSV files (per language):"
find $RESULTS_DIR -name "*_length_analysis.csv" | sort
echo ""
echo "📊 Aggregated table files (per model):"
find $RESULTS_DIR -name "aggregated_length_results_*.csv" | sort
echo ""
echo "📋 Key findings to report:"
echo "1. Text length distribution (now equal 1/3 splits per language)"
echo "2. Watermark detection accuracy by text length (percentile-based)"
echo "3. AUC scores comparing short vs medium vs long texts (reliable with equal samples)"
echo "4. Language detection performance variations by text length"
echo "5. Language-specific length thresholds and their impact on detection"
echo ""
echo "🔍 Check the aggregated tables for:"
echo "  - Per-model tables with languages as rows, length categories as columns"
echo "  - AUC and accuracy scores for each length category"
echo "  - Sample distribution across length categories"