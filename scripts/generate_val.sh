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
    "meta-llama/Llama-3.2-1B"
    "LLaMAX/LLaMAX3-8B"
)
MODEL_ABBRS=(
    "aya-23-8B"
    "llama-3.2-1B"
    "llamax3-8B"
)

# Settings
WATERMARK_METHODS=("kgw")
SEEDS=(0)
SUPPORTED_LANGS=(
    "af" "sq" "am" "ar" "hy" "as" "ay" "az" "bm" "eu"
    "be" "bn" "bho" "bs" "bg" "ca" "ceb" "ny" "zh-CN" "zh-TW"
    "co" "hr" "cs" "da" "dv" "doi" "nl" "en" "eo" "et"
    "ee" "tl" "fi" "fr" "fy" "gl" "ka" "de" "el" "gn"
    "gu" "ht" "ha" "haw" "iw" "hi" "hmn" "hu" "is" "ig"
    "ilo" "id" "ga" "it" "ja" "jw" "kn" "kk" "km" "rw"
    "gom" "ko" "kri" "ku" "ckb" "ky" "lo" "la" "lv" "ln"
    "lt" "lg" "lb" "mk" "mai" "mg" "ms" "ml" "mt" "mi"
    "mr" "mni-Mtei" "lus" "mn" "my" "ne" "no" "or" "om" "ps"
    "fa" "pl" "pt" "pa" "qu" "ro" "ru" "sm" "sa" "gd"
    "nso" "sr" "st" "sn" "sd" "si" "sk" "sl" "so" "es"
    "su" "sw" "sv" "tg" "ta" "tt" "te" "th" "ti" "ts"
    "tr" "tk" "ak" "uk" "ur" "ug" "uz" "vi" "cy" "xh"
    "yi" "yo" "zu"
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


            for TGT_LANG in "${SUPPORTED_LANGS[@]}"; do
                echo "🌍 Translating and detecting for $TGT_LANG"

                # Translation of validation text
                # echo "🌐 Translating validation text to $TGT_LANG"
                # python3 "$ATTACK_DIR/google_translate.py" \
                #     --input_file "$DATA_DIR/dataset/mc4/mc4.en.val.jsonl" \
                #     --output_file "$DATA_DIR/dataset/mc4/mc4.${TGT_LANG}.val.jsonl" \
                #     --translation_part response \
                #     --src_lang en \
                #     --tgt_lang "$TGT_LANG"

                echo "🌐 Detecting on $TGT_LANG validation set"
                python3 "$WORK_DIR/detect.py" \
                    --base_model "$MODEL_NAME" \
                    --seed "$SEED" \
                    --detect_file "$DATA_DIR/dataset/mc4/mc4.${TGT_LANG}.val.jsonl" \
                    --output_file "$OUT_DIR/mc4.${TGT_LANG}.val.z_score.jsonl" \
                    $WATERMARK_FLAGS
            done
        done
    done
done