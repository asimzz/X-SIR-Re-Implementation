#!/bin/bash

# Exit on any error or unset variable
set -e
set -u

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"

# Model configuration
TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE=32

# Model names and abbreviations
MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)
MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

# Settings
WATERMARK_METHODS=("kgw" "xsir")
SEEDS=(0)

# Target languages for BO-enhanced STEAM evaluation
# Format: "ISO_639_3:ISO_639_1" (for URIEL:for filenames)
# ISO 639-3 (3-letter) is used for URIEL genetic distances
# ISO 639-1 (2-letter) is used for translated file naming
TGT_LANGS=(
    # High-resource languages
    # "fra:fr"   # French
    "deu:de"   # German
    # "ita:it"   # Italian
    "spa:es"   # Spanish
    # "por:pt"   # Portuguese
    
    # Medium-resource languages
    # "pol:pl"   # Polish
    # "nld:nl"   # Dutch
    # "rus:ru"   # Russian
    "hin:hi"   # Hindi
    "kor:ko"   # Korean
    # "jpn:ja"   # Japanese
    
    # Low-resource languages
    "ben:bn"   # Bengali
    "pes:fa"   # Persian (note: was 'fas' but URIEL uses 'pes')
    # "vie:vi"   # Vietnamese
    # "heb:he"   # Hebrew
    # "ukr:uk"   # Ukrainian
    # "tam:ta"   # Tamil
)

# BO Configuration with URIEL Genetic Distances
N_INITIAL=3          # Number of random initial language samples
MAX_EVALUATIONS=8    # Maximum total language evaluations per text (3 initial + 5 BO)
N_SAMPLES=50         # Number of text samples to evaluate (subset for testing)
LANGUAGES_FILE="$WORK_DIR/all_languages.txt"  # 100 URIEL-supported languages

# Validation check
if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "❌ Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

# Check if languages file exists
if [ ! -f "$LANGUAGES_FILE" ]; then
    echo "❌ Languages file not found: $LANGUAGES_FILE"
    echo "   Please ensure all_languages.txt is in the working directory"
    exit 1
fi

echo "=========================================="
echo "BO-Enhanced STEAM Evaluation with URIEL"
echo "=========================================="
echo "Using URIEL genetic distances (NO CLUSTERS)"
echo "Intermediate languages: $(wc -l < "$LANGUAGES_FILE")"
echo "Initial samples: $N_INITIAL"
echo "Max evaluations per text: $MAX_EVALUATIONS"
echo "Test samples: $N_SAMPLES"
echo "=========================================="
echo ""

# Main evaluation loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    echo "📊 Processing model: $MODEL_ABBR"
    echo ""

    for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            # Set output directory
            OUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
            mkdir -p "$OUT_DIR"

            # Set watermark-specific flags
            if [ "$WATERMARK_METHOD" == "kgw" ]; then
                WATERMARK_FLAGS="--watermark_method kgw"
            elif [ "$WATERMARK_METHOD" == "xsir" ]; then
                MAPPING_FILE="$DATA_DIR/mapping/xsir/mapping_${MODEL_ABBR}_en_mc4.json"
                WATERMARK_FLAGS="--watermark_method xsir --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_FILE"
            else
                echo "❌ Unknown watermark method: $WATERMARK_METHOD"
                exit 1
            fi

            # Check if watermarked data exists
            if [ ! -f "$OUT_DIR/mc4.en.mod.jsonl" ]; then
                echo "⚠️  Watermarked data not found: $OUT_DIR/mc4.en.mod.jsonl"
                echo "   Skipping $MODEL_ABBR/$WATERMARK_METHOD/seed$SEED"
                echo ""
                continue
            fi

            # BO-Enhanced STEAM evaluation for each target language
            for LANG_PAIR in "${TGT_LANGS[@]}"; do
                # Split the pair into ISO3 and ISO1 codes
                ISO3_CODE="${LANG_PAIR%%:*}"  # Everything before ':'
                ISO1_CODE="${LANG_PAIR##*:}"  # Everything after ':'
                
                echo "🔍 Evaluating: $WATERMARK_METHOD / $ISO3_CODE ($ISO1_CODE) / BO-STEAM"

                # Translated files use ISO 639-1 codes in filenames
                TRANSLATED_FILE="$OUT_DIR/mc4.en-${ISO1_CODE}.mod.jsonl"
                if [ ! -f "$TRANSLATED_FILE" ]; then
                    echo "⚠️  Translated data not found: $TRANSLATED_FILE"
                    echo "   Skipping..."
                    echo ""
                    continue
                fi

                # Output files use ISO 639-3 codes for consistency with URIEL
                BO_OUTPUT_FILE="$OUT_DIR/mc4.en-${ISO3_CODE}.bo_steam_results.jsonl"
                BO_SUMMARY_FILE="$OUT_DIR/mc4.en-${ISO3_CODE}.bo_steam_summary.json"

                # Run BO-enhanced STEAM evaluation with URIEL genetic distances
                # Note: Pass ISO 639-3 code to evaluation script for URIEL compatibility
                python3 "$WORK_DIR/evaluate_bo_steam.py" \
                    --base_model "$MODEL_NAME" \
                    --input_file "$TRANSLATED_FILE" \
                    --output_file "$BO_OUTPUT_FILE" \
                    --summary_file "$BO_SUMMARY_FILE" \
                    --target_lang "$ISO3_CODE" \
                    --n_initial "$N_INITIAL" \
                    --max_evaluations "$MAX_EVALUATIONS" \
                    --n_samples "$N_SAMPLES" \
                    --languages_file "$LANGUAGES_FILE" \
                    --seed "$SEED" \
                    $WATERMARK_FLAGS

                echo "✅ Results saved:"
                echo "   - Detailed: $BO_OUTPUT_FILE"
                echo "   - Summary: $BO_SUMMARY_FILE"
                echo ""

            done

        done
    done
done