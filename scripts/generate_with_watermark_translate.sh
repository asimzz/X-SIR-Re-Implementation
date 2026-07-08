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
    # "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
    # "LLaMAX/LLaMAX3-8B"
)
MODEL_ABBRS=(
    # "llama-3.2-1B"
    "aya-23-8B"
    # "llamax3-8B"
)

# Settings
WATERMARK_METHODS=("kgw")
SEEDS=(0)
TGT_LANGS=(
    # High-resource languages
    "en" "de" "it" "es" "pt"
    # Medium-resource languages
    "pl" "nl" "ru" "hi" "ko" "ja"
    # Low-resource languages
    "bn" "fa" "vi" "iw" "uk" "ta"
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

            # # Step 1: Generate watermarked data
            python3 "$WORK_DIR/gen.py" \
                --base_model "$MODEL_NAME" \
                --fp16 \
                --batch_size "$BATCH_SIZE" \
                --seed "$SEED" \
                --input_file "$DATA_DIR/dataset/mc4/mc4.fr.jsonl" \
                --output_file "$OUT_DIR/mc4.fr.mod.jsonl" \
                $WATERMARK_FLAGS

            # Step 2: Detect watermark in French
            python3 "$WORK_DIR/detect.py" \
                --base_model "$MODEL_NAME" \
                --seed "$SEED" \
                --detect_file "$OUT_DIR/mc4.fr.mod.jsonl" \
                --output_file "$OUT_DIR/mc4.fr.mod.z_score.jsonl" \
                $WATERMARK_FLAGS

            # Step 3: Translation & detection for each target language
            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo "🌍 Translating and detecting for $TGT_LANG"

                # Translation attack
                python3 "$ATTACK_DIR/google_translate.py" \
                    --input_file "$OUT_DIR/mc4.fr.mod.jsonl" \
                    --output_file "$OUT_DIR/mc4.fr-${TGT_LANG}.mod.jsonl" \
                    --translation_part response \
                    --src_lang fr \
                    --tgt_lang "$TGT_LANG"

                # Detect on translated output
                python3 "$WORK_DIR/detect.py" \
                    --base_model "$MODEL_NAME" \
                    --seed "$SEED" \
                    --detect_file "$OUT_DIR/mc4.fr-${TGT_LANG}.mod.jsonl" \
                    --output_file "$OUT_DIR/mc4.fr-${TGT_LANG}.mod.z_score.jsonl" \
                    $WATERMARK_FLAGS
            done
        done
    done
done