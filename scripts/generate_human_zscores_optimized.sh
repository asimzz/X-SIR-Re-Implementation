#!/bin/bash

# Exit on any error or unset variable
set -e
set -u

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"
ATTACK_DIR="$WORK_DIR/attack"
MAPPING_DIR="$DATA_DIR/mapping/xsir"

# Path to optimized pivot languages CSV
PIVOT_CSV="$WORK_DIR/pivot_language_extraction/max_distance_hierarchical_clean.csv"

# Model configuration
TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE=32

# Model names and abbreviations
MODEL_NAMES=(
    # "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)
MODEL_ABBRS=(
    # "llama-3.2-1B"
    "aya-23-8B"
)

# Settings
WATERMARK_METHODS=("kgw")
SEEDS=(0)
TGT_LANGS=(
    # High-resource languages
    "fr" # French
    "de" # German
    "it" # Italian
    "es" # Spanish
    "pt" # Portuguese
    # Medium-resource languages
    "pl" # Polish
    # "nl" # Dutch
    # "ru" # Russian
    # "hi" # Hindi
    # "ko" # Korean
    # "ja" # Japanese
    # Low-resource languages
    # "bn" # Bengali
    # "fa" # Persian
    # "vi" # Vietnamese
    # "iw" # Hebrew
    # "uk" # Ukrainian
    # "ta" # Tamil
)

# Function to get optimized pivot languages for a target language
get_pivot_languages() {
    local target_lang="$1"

    # Check if CSV file exists
    if [ ! -f "$PIVOT_CSV" ]; then
        echo "❌ Error: Optimized pivot languages file not found: $PIVOT_CSV"
        exit 1
    fi

    # Use Python to properly parse the CSV with quoted lists
    local pivot_languages=$(python3 -c "
import csv
import ast

target = '$target_lang'
with open('$PIVOT_CSV', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if row[0] == target:
            # Parse the list string
            lang_list = ast.literal_eval(row[1])
            print(' '.join(lang_list))
            break
    else:
        exit(1)
")

    if [ $? -ne 0 ] || [ -z "$pivot_languages" ]; then
        echo "❌ Error: Could not parse pivot languages for target language: $target_lang"
        exit 1
    fi

    echo "$pivot_languages"
}

# Function to convert 3-letter to 2-letter language codes (for Google Translate API)
convert_to_2_letter() {
    local lang_3="$1"
    case "$lang_3" in
        "fin") echo "fi" ;;
        "hun") echo "hu" ;;
        "tur") echo "tr" ;;
        "uig") echo "ug" ;;  # Uyghur
        "spa") echo "es" ;;
        "fra") echo "fr" ;;
        "eng") echo "en" ;;
        "deu") echo "de" ;;
        "ita") echo "it" ;;
        "por") echo "pt" ;;
        "pol") echo "pl" ;;
        "nld") echo "nl" ;;
        "rus") echo "ru" ;;
        "hin") echo "hi" ;;
        "kor") echo "ko" ;;
        "jpn") echo "ja" ;;
        "ben") echo "bn" ;;
        "fas") echo "fa" ;;
        "vie") echo "vi" ;;
        "heb") echo "iw" ;;
        "ukr") echo "uk" ;;
        "tam") echo "ta" ;;
        *) echo "$lang_3" ;;  # fallback
    esac
}

# Validate model list lengths
if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "❌ MODEL_NAMES and MODEL_ABBRS length mismatch."
    exit 1
fi

echo "🧬 OPTIMIZED HUMAN Z-SCORE GENERATION"
echo "📊 Using optimized pivot languages from genetic distance optimization"
echo "📊 Pivot languages source: $PIVOT_CSV"
echo "🎯 Processing human text data with adaptive pivot selection"
echo ""

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            echo "▶️ Running $WATERMARK_METHOD (seed=$SEED) on $MODEL_NAME"

            MAPPING_FILE="$MAPPING_DIR/300_mapping_${MODEL_ABBR}_seed${SEED}.json"
            OUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}"
            mkdir -p "$OUT_DIR"

            if [ $WATERMARK_METHOD == "kgw" ]; then
                WATERMARK_FLAGS="--watermark_method kgw"
            elif [ "$WATERMARK_METHOD" == "xsir" ]; then
                WATERMARK_FLAGS="--watermark_method xsir --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_FILE"
            else
                echo "❌ Unknown watermark method: $WATERMARK_METHOD"
                exit 1
            fi

            # Step 1: Generate human z-scores for English
            # echo "📝 Generating human z-scores for English..."
            # python3 "$WORK_DIR/detect.py" \
            #     --base_model "$MODEL_NAME" \
            #     --seed "$SEED" \
            #     --detect_file "$DATA_DIR/dataset/mc4/mc4.en.jsonl" \
            #     --output_file "$OUT_DIR/mc4.en.hum.z_score.jsonl" \
            #     $WATERMARK_FLAGS

            # Step 2: Translation & detection for each target language
            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo ""
                echo "🌍 Processing human data for target language: $TGT_LANG"
                echo "-----------------------------------------------"

                # Get optimized pivot languages for this target
                PIVOT_LANGUAGES_3LETTER=$(get_pivot_languages "$TGT_LANG")
                echo "🧬 Optimized pivot languages (3-letter): $PIVOT_LANGUAGES_3LETTER"

                # Convert to array
                read -r -a PIVOT_ARRAY_3 <<< "$PIVOT_LANGUAGES_3LETTER"

                # Convert to 2-letter codes for Google Translate
                PIVOT_ARRAY_2=()
                PIVOT_DISPLAY=""
                for pivot_3 in "${PIVOT_ARRAY_3[@]}"; do
                    pivot_2=$(convert_to_2_letter "$pivot_3")
                    PIVOT_ARRAY_2+=("$pivot_2")
                    PIVOT_DISPLAY="${PIVOT_DISPLAY}${pivot_3}(${pivot_2}) "
                done

                echo "🗣️  Using pivot languages: $PIVOT_DISPLAY"

                # Translation of human text to target language
                # echo "🌐 Translating human text: English -> $TGT_LANG..."
                # python3 "$ATTACK_DIR/google_translate.py" \
                #     --input_file "$DATA_DIR/dataset/mc4/mc4.en.jsonl" \
                #     --output_file "$OUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl" \
                #     --translation_part response \
                #     --src_lang en \
                #     --tgt_lang "$TGT_LANG"

                # echo "🔍 Detecting watermark in translated human text (${TGT_LANG})..."
                # # Detect on translated human text
                # python3 "$WORK_DIR/detect.py" \
                #     --base_model "$MODEL_NAME" \
                #     --seed "$SEED" \
                #     --detect_file "$OUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl" \
                #     --output_file "$OUT_DIR/mc4.en-${TGT_LANG}.hum.z_score.jsonl" \
                #     $WATERMARK_FLAGS

                # Back-translation using OPTIMIZED pivot languages
                for pivot_lang in "${PIVOT_ARRAY_2[@]}"; do
                    echo "🔁 OPTIMIZED back-translation human: $TGT_LANG -> $pivot_lang"

                    # Back-translation of human data
                    python3 "$ATTACK_DIR/google_translate.py" \
                        --input_file "$OUT_DIR/mc4.en-$TGT_LANG.hum.jsonl" \
                        --output_file "$OUT_DIR/mc4.$TGT_LANG-$pivot_lang-back.hum.jsonl" \
                        --translation_part response \
                        --src_lang "$TGT_LANG" \
                        --tgt_lang "$pivot_lang"

                    # echo "🔍 Detecting watermark in back-translated human text (${TGT_LANG} -> ${pivot_lang})..."
                    # # Detect watermark in back-translation
                    # python3 "$WORK_DIR/detect.py" \
                    #     --base_model "$MODEL_NAME" \
                    #     --seed "$SEED" \
                    #     --detect_file "$OUT_DIR/mc4.$TGT_LANG-$pivot_lang-back.hum.jsonl" \
                    #     --output_file "$OUT_DIR/mc4.$TGT_LANG-$pivot_lang-back.hum.z_score.jsonl" \
                        # $WATERMARK_FLAGS
                done

                echo "✅ Completed optimized human text processing for $TGT_LANG"
                echo "📊 Used ${#PIVOT_ARRAY_2[@]} optimized pivot languages instead of fixed languages"
            done

            echo ""
            echo "🎉 Completed all optimized human z-score generation for $WATERMARK_METHOD (seed=$SEED)"
        done
    done
done

echo ""
echo "🏆 OPTIMIZED HUMAN Z-SCORE GENERATION COMPLETE!"
echo "🧬 Used genetic distance-optimized pivot languages for each target"
echo "📈 Human text data processed with adaptive pivot selection"
echo "🔄 Next: Use these z-scores for STEAM evaluation against watermarked text"