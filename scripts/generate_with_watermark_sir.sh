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
MAPPING_DIR="$DATA_DIR/mapping"

# Model configuration
TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE=32

# Model names and abbreviations
MODEL_NAMES=(
    "CohereForAI/aya-23-8B"
    # "meta-llama/Llama-3.2-1B"
    # "LLaMAX/LLaMAX3-8B"
)

MODEL_ABBRS=(
    "aya-23-8B"
    # "llama-3.2-1B"
    # "llamax3-8B"
)

SEEDS=(0)

# Settings
WATERMARK_METHODS=("sir")
ORG_LANG="en"
PVT_LANGS=(
    "de"
    "fr"
    # "zh"
    # "ja"
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
            echo "▶️ Running $WATERMARK_METHOD on $MODEL_NAME"

            MAPPING_FILE="$MAPPING_DIR/$WATERMARK_METHOD/300_mapping_${MODEL_ABBR}.json"
            OUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}"
            mkdir -p "$OUT_DIR"

            if [ $WATERMARK_METHOD == "kgw" ]; then
                WATERMARK_FLAGS="--watermark_method kgw"
            elif [ "$WATERMARK_METHOD" == "sir" ] || [ "$WATERMARK_METHOD" == "xsir" ]; then
                WATERMARK_FLAGS="--watermark_method xsir --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_FILE"
            else
                echo "❌ Unknown watermark method: $WATERMARK_METHOD"
                exit 1
            fi


            for PVT_LANG in "${PVT_LANGS[@]}"; do
            echo "🌐 Processing language pair: $ORG_LANG ➝ $PVT_LANG"
            python3 "$ATTACK_DIR/google_translate.py" \
                --input_file "$OUT_DIR/mc4.$ORG_LANG.mod.jsonl" \
                --output_file "$OUT_DIR/mc4.$ORG_LANG-${PVT_LANG}-cwra.jsonl" \
                --translation_part prompt \
                --src_lang "$ORG_LANG" \
                --tgt_lang "$PVT_LANG"

            echo "🧬 Generating watermark on translated prompts"
                python3 "$WORK_DIR/gen.py" \
                    --base_model "$MODEL_NAME" \
                    --fp16 \
                    --batch_size "$BATCH_SIZE" \
                    --input_file "$OUT_DIR/mc4.$ORG_LANG-$PVT_LANG-cwra.jsonl" \
                    --output_file "$OUT_DIR/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.jsonl" \
                    $WATERMARK_FLAGS

                echo "🔍 Detecting watermark post-generation"
                python3 "$WORK_DIR/detect.py" \
                    --base_model "$MODEL_NAME" \
                    --detect_file "$OUT_DIR/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.jsonl" \
                    --output_file "$OUT_DIR/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.z_score.jsonl" \
                    $WATERMARK_FLAGS

                echo "🔄 CWRA: back-translating response $PVT_LANG ➝ $ORG_LANG"
                python3 "$ATTACK_DIR/google_translate.py" \
                    --input_file "$OUT_DIR/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.jsonl" \
                    --output_file "$OUT_DIR/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.jsonl" \
                    --src_lang "$PVT_LANG" \
                    --tgt_lang "$ORG_LANG" \
                    --translation_part response

                python3 "$WORK_DIR/detect.py" \
                    --base_model "$MODEL_NAME" \
                    --detect_file "$OUT_DIR/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.jsonl" \
                    --output_file "$OUT_DIR/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.z_score.jsonl" \
                    $WATERMARK_FLAGS
            done
        done
    done
done