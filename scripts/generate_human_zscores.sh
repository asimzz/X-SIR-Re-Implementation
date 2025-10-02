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

# Model configuration
TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE=32

# Model names and abbreviations
MODEL_NAMES=(
    "CohereForAI/aya-23-8B"
)
MODEL_ABBRS=(
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
    "nl" # Dutch
    "ru" # Russian
    "hi" # Hindi
    "ko" # Korean
    "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    "vi" # Vietnamese
    "iw" # Hebrew
    "uk" # Ukrainian
    "ta" # Tamil
)

PVT_LANGS=(
    "de" # German
    "ko" # Korean
    "bn" # Bengali
)

ORG_LANGS=(
    "en" # English
    # High-resource languages
    "fr" # French
    "de" # German
    "it" # Italian
    "es" # Spanish
    "pt" # Portuguese
    # Medium-resource languages
    "pl" # Polish
    "nl" # Dutch
    "ru" # Russian
    "hi" # Hindi
    "ko" # Korean
    "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    "vi" # Vietnamese
    "iw" # Hebrew
    "uk" # Ukrainian
    "ta" # Tamil
)

# Validate model list lengths
if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "❌ MODEL_NAMES and MODEL_ABBRS length mismatch."
    exit 1
fi

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


            # # Step 1: Generate human text
            # echo "📝 Generating human text for en"
            # python3 "$WORK_DIR/detect.py" \
            #     --base_model "$MODEL_NAME" \
            #     --seed "$SEED" \
            #     --detect_file "$DATA_DIR/dataset/mc4/mc4.en.jsonl" \
            #     --output_file "$OUT_DIR/mc4.en.hum.z_score.jsonl" \
            #     $WATERMARK_FLAGS

            # Step 3: Translation & detection for each target language
            for TGT_LANG in "${TGT_LANGS[@]}"; do
                # echo "🌍 Translating and detecting for $TGT_LANG"

                # # Translation of human text
                # echo "🌐 Translating human text to $TGT_LANG"
                # python3 "$ATTACK_DIR/google_translate.py" \
                #     --input_file "$DATA_DIR/dataset/mc4/mc4.en.jsonl" \
                #     --output_file "$OUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl" \
                #     --translation_part response \
                #     --src_lang en \
                #     --tgt_lang "$TGT_LANG"

                # echo "🔍 Detecting watermark in translated human text"
                # # Detect on translated human text
                # python3 "$WORK_DIR/detect.py" \
                #     --base_model "$MODEL_NAME" \
                #     --seed "$SEED" \
                #     --detect_file "$OUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl" \
                #     --output_file "$OUT_DIR/mc4.en-${TGT_LANG}.hum.z_score.jsonl" \
                #     $WATERMARK_FLAGS

                # Pivoting attack (using multiple pivot languages)
                for PVT_LANG in "${PVT_LANGS[@]}"; do
                    if [ "$PVT_LANG" == "$TGT_LANG" ]; then
                        continue
                    fi
                    # Pivot translation
                    echo "🔀 Pivot translation $TGT_LANG -> $PVT_LANG"
                    python3 "$ATTACK_DIR/google_translate.py" \
                        --input_file "$OUT_DIR/mc4.en-$TGT_LANG.hum.jsonl" \
                        --output_file "$OUT_DIR/mc4.$TGT_LANG-$PVT_LANG-pivot.hum.jsonl" \
                        --translation_part response \
                        --src_lang "$TGT_LANG" \
                        --tgt_lang "$PVT_LANG"
                    # Detect on pivoted human text
                    python3 "$WORK_DIR/detect.py" \
                        --base_model "$MODEL_NAME" \
                        --seed "$SEED" \
                        --detect_file "$OUT_DIR/mc4.$TGT_LANG-$PVT_LANG-pivot.hum.jsonl" \
                        --output_file "$OUT_DIR/mc4.$TGT_LANG-$PVT_LANG-pivot.hum.z_score.jsonl" \
                        $WATERMARK_FLAGS

                    for ORG_LANG in "${ORG_LANGS[@]}"; do
                        if [ "$ORG_LANG" == "$PVT_LANG" ]; then
                            continue
                        fi
                        # Back-translation
                        echo "🔁 Back translation $PVT_LANG -> $ORG_LANG"
                        python3 "$ATTACK_DIR/google_translate.py" \
                            --input_file "$OUT_DIR/mc4.$TGT_LANG-$PVT_LANG-pivot.hum.jsonl" \
                            --output_file "$OUT_DIR/mc4.$TGT_LANG-$PVT_LANG-$ORG_LANG-back.hum.jsonl" \
                            --translation_part response \
                            --src_lang "$PVT_LANG" \
                            --tgt_lang "$ORG_LANG"

                        python3 "$WORK_DIR/detect.py" \
                            --base_model "$MODEL_NAME" \
                            --seed "$SEED" \
                            --detect_file "$OUT_DIR/mc4.$TGT_LANG-$PVT_LANG-$ORG_LANG-back.hum.jsonl" \
                            --output_file "$OUT_DIR/mc4.$TGT_LANG-$PVT_LANG-$ORG_LANG-back.hum.z_score.jsonl" \
                            $WATERMARK_FLAGS
                    done
                done
            done
        done
    done
done