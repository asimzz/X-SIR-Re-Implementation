#!/bin/bash

# STEAM BO Detection Pipeline Runner
# Runs per-text Bayesian Optimization to find optimal pivot languages
# for watermark detection after translation attacks

set -e  # Exit on any error

# Configuration
BASE_MODEL="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
DEFAULT_WATERMARK_METHOD="kgw"  # Default method, can be overridden
N_INITIAL=3
MAX_EVALUATIONS=8
NUM_TEXTS=500
RANDOM_STATE=42

# Directories
INPUT_BASE_DIR="gen"
OUTPUT_BASE_DIR="results/steam_bo"
LOG_DIR="logs/steam_bo"

# Create necessary directories
mkdir -p $OUTPUT_BASE_DIR
mkdir -p $LOG_DIR

# Target languages to process
TARGET_LANGS=("fr" "de" "es" "it" "pt" "nl" "pl" "ru" "hi" "ko" "ja" "bn" "fa" "vi" "iw" "uk" "ta")

echo "========================================"
echo "STEAM BO Detection Pipeline"
echo "========================================"
echo "Base Model: $BASE_MODEL"
echo "Watermark Method: $WATERMARK_METHOD"
echo "Initial Pivots: $N_INITIAL"
echo "Max Evaluations: $MAX_EVALUATIONS"
echo "Texts per Language: $NUM_TEXTS"
echo "Target Languages: ${TARGET_LANGS[*]}"
echo "========================================"

# Function to run STEAM BO for a single target language
run_steam_bo() {
    local TGT_LANG=$1
    local WATERMARK_METHOD=${2:-$DEFAULT_WATERMARK_METHOD}  # Use provided method or default
    local METHOD_DIR=""

    # Set method-specific directory
    if [[ "$WATERMARK_METHOD" == "kgw" ]]; then
        METHOD_DIR="kgw_seed0"
    elif [[ "$WATERMARK_METHOD" == "xsir" ]]; then
        METHOD_DIR="xsir"
    else
        METHOD_DIR="$WATERMARK_METHOD"
    fi

    local INPUT_DIR="${INPUT_BASE_DIR}/${MODEL_ABBR}/${METHOD_DIR}"
    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_ABBR}_${WATERMARK_METHOD}/${TGT_LANG}"
    local LOG_FILE="${LOG_DIR}/steam_bo_${WATERMARK_METHOD}_${TGT_LANG}.log"

    echo ""
    echo "----------------------------------------"
    echo "Processing Target Language: $TGT_LANG"
    echo "----------------------------------------"
    echo "Input Dir: $INPUT_DIR"
    echo "Output Dir: $OUTPUT_DIR"
    echo "Log File: $LOG_FILE"

    # Check if input files exist
    MOD_FILE="${INPUT_DIR}/mc4.en-${TGT_LANG}.mod.jsonl"
    HUM_FILE="${INPUT_DIR}/mc4.en-${TGT_LANG}.hum.jsonl"
    VAL_FILE="${INPUT_DIR}/mc4.en-${TGT_LANG}.val.jsonl"

    if [[ ! -f "$MOD_FILE" ]]; then
        echo "ERROR: Missing watermarked file: $MOD_FILE"
        return 1
    fi

    if [[ ! -f "$HUM_FILE" ]]; then
        echo "ERROR: Missing human file: $HUM_FILE"
        return 1
    fi

    if [[ ! -f "$VAL_FILE" ]]; then
        echo "ERROR: Missing validation file: $VAL_FILE"
        return 1
    fi

    echo "✓ Input files verified"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Set up method-specific parameters
    STEAM_ARGS="--base_model $BASE_MODEL --tgt_lang $TGT_LANG --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --watermark_method $WATERMARK_METHOD --n_initial $N_INITIAL --max_evaluations $MAX_EVALUATIONS --num_texts $NUM_TEXTS --random_state $RANDOM_STATE"

    if [[ "$WATERMARK_METHOD" == "xsir" ]]; then
        # XSIR specific parameters
        TRANSFORM_MODEL="data/model/transform_model_x-sbert_10K.pth"
        EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
        MAPPING_FILE="data/mapping/xsir/300_mapping_${MODEL_ABBR}.json"

        # Check if XSIR files exist
        if [[ ! -f "$TRANSFORM_MODEL" ]]; then
            echo "ERROR: XSIR transform model not found: $TRANSFORM_MODEL"
            return 1
        fi

        if [[ ! -f "$MAPPING_FILE" ]]; then
            echo "ERROR: XSIR mapping file not found: $MAPPING_FILE"
            return 1
        fi

        STEAM_ARGS="$STEAM_ARGS --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_FILE"
    fi

    # Run STEAM BO optimization
    echo "Starting STEAM BO optimization ($WATERMARK_METHOD)..."
    echo "Note: Validation baselines will be computed on-the-fly if not present"

    python3 steam_bo_detector.py $STEAM_ARGS 2>&1 | tee "$LOG_FILE"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "✓ STEAM BO optimization completed successfully for $TGT_LANG"

        # Check if results files were created
        RESULTS_FILE="${OUTPUT_DIR}/steam_bo_results_${TGT_LANG}.json"
        SUMMARY_FILE="${OUTPUT_DIR}/steam_bo_summary_${TGT_LANG}.json"

        if [[ -f "$RESULTS_FILE" && -f "$SUMMARY_FILE" ]]; then
            echo "✓ Results saved:"
            echo "  - $RESULTS_FILE"
            echo "  - $SUMMARY_FILE"

            # Extract summary statistics
            if command -v jq >/dev/null 2>&1; then
                echo ""
                echo "Summary Statistics:"
                jq -r '.overall_stats |
                "  Texts Processed: " + (.num_texts_processed | tostring) +
                "\n  Mean Best Score: " + (.mean_best_score | tostring) +
                "\n  Std Best Score: " + (.std_best_score | tostring) +
                "\n  Avg Evaluations: " + (.avg_evaluations_per_text | tostring)' "$SUMMARY_FILE"

                echo ""
                echo "Top Pivot Languages:"
                jq -r '.overall_stats.pivot_language_distribution | to_entries | sort_by(-.value) | .[:5] | .[] | "  " + .key + ": " + (.value | tostring) + " texts"' "$SUMMARY_FILE"
            fi
        else
            echo "⚠ Warning: Results files not found"
        fi
    else
        echo "✗ STEAM BO optimization failed for $TGT_LANG"
        return 1
    fi
}


# Main execution
main() {
    local START_TIME=$(date +%s)
    local FAILED_LANGS=()
    local SUCCESSFUL_LANGS=()

    echo "Starting STEAM BO pipeline at $(date)"

    # Process each target language
    for TGT_LANG in "${TARGET_LANGS[@]}"; do
        if run_steam_bo "$TGT_LANG" "$DEFAULT_WATERMARK_METHOD"; then
            SUCCESSFUL_LANGS+=("$TGT_LANG")
        else
            FAILED_LANGS+=("$TGT_LANG")
            echo "⚠ Continuing with next language..."
        fi
    done

    # Summary
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "========================================"
    echo "STEAM BO Pipeline Complete"
    echo "========================================"
    echo "Total Duration: $DURATION seconds"
    echo "Successful Languages (${#SUCCESSFUL_LANGS[@]}): ${SUCCESSFUL_LANGS[*]}"

    if [[ ${#FAILED_LANGS[@]} -gt 0 ]]; then
        echo "Failed Languages (${#FAILED_LANGS[@]}): ${FAILED_LANGS[*]}"
    fi

    echo ""
    echo "Results Directory: $OUTPUT_BASE_DIR"
    echo "Logs Directory: $LOG_DIR"

    # Create overall summary
    OVERALL_SUMMARY="${OUTPUT_BASE_DIR}/pipeline_summary.json"
    cat > "$OVERALL_SUMMARY" << EOF
{
    "pipeline_run": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "duration_seconds": $DURATION,
        "base_model": "$BASE_MODEL",
        "watermark_method": "$WATERMARK_METHOD",
        "configuration": {
            "n_initial": $N_INITIAL,
            "max_evaluations": $MAX_EVALUATIONS,
            "num_texts": $NUM_TEXTS,
            "random_state": $RANDOM_STATE
        }
    },
    "results": {
        "successful_languages": [$(printf '"%s",' "${SUCCESSFUL_LANGS[@]}" | sed 's/,$//')]",
        "failed_languages": [$(printf '"%s",' "${FAILED_LANGS[@]}" | sed 's/,$//')]",
        "success_rate": $(echo "scale=3; ${#SUCCESSFUL_LANGS[@]} / ${#TARGET_LANGS[@]}" | bc -l)
    }
}
EOF

    echo "Overall summary saved to: $OVERALL_SUMMARY"
}

# Command line argument handling
if [[ $# -eq 0 ]]; then
    # Run main pipeline
    main
elif [[ $1 == "single" && $# -eq 2 ]]; then
    # Run single language
    TGT_LANG=$2
    echo "Running STEAM BO for single language: $TGT_LANG (KGW)"
    run_steam_bo "$TGT_LANG" "kgw"
elif [[ $1 == "xsir" && $# -eq 2 ]]; then
    # Run XSIR for single language
    TGT_LANG=$2
    echo "Running STEAM BO (XSIR) for single language: $TGT_LANG"
    run_steam_bo "$TGT_LANG" "xsir"
elif [[ $1 == "help" || $1 == "-h" || $1 == "--help" ]]; then
    echo "Usage:"
    echo "  $0                    # Run full pipeline for all languages"
    echo "  $0 single <lang>      # Run for single language (KGW)"
    echo "  $0 xsir <lang>        # Run for single language (XSIR)"
    echo "  $0 help               # Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                    # Process all languages"
    echo "  $0 single fr          # Process only French"
    echo "  $0 xsir de            # Process German with XSIR"
else
    echo "Invalid arguments. Use '$0 help' for usage information."
    exit 1
fi