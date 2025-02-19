set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack

# Parameters for SIR/X-SIR
MAPPING_DIR=$WORK_DIR/data/mapping
TRANSFORM_MODEL=$WORK_DIR/data/model/transform_model_x-sbert_10K.pth
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

BATCH_SIZE=8

MODEL_NAMES=(
    "bigscience/bloom-7b1"
    "CohereForAI/aya-23-8B"
    # "meta-llama/Llama-2-7b-hf"
    # "baichuan-inc/Baichuan2-7B-Base"
    # "baichuan-inc/Baichuan-7B"

)

MODEL_ABBRS=(
    "bloom-7b1"
    "aya-23-8B"
    # "llama2-7b"
    # "baichuan2-7b"
    # "baichuan-7b"
)

WATERMARK_METHODS=(
    "kgw"
    # "xsir"
)

ORG_LANG="en"
PVT_LANGS= (
    "ar"
    "tr"
    "de"
    "fr"
    "zh"
    "ja"
)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
        echo "Generating with watermark for $MODEL_NAME using $WATERMARK_METHOD"

        if [ $WATERMARK_METHOD == "kgw" ]; then
            WATERMARK_METHOD_FLAG="--watermark_method kgw"
        elif [ $WATERMARK_METHOD == "sir" ] || [ $WATERMARK_METHOD == "xsir" ]; then
            WATERMARK_METHOD_FLAG="--watermark_method xsir  --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_DIR/$WATERMARK_METHOD/300_mapping_$MODEL_ABBR.json"
        else
            echo "Unknown watermark method: $WATERMARK_METHOD"
            exit 1
        fi

        for PVT_LANG in "${PVT_LANGS[@]}"; do
            echo "Translating prompts $ORG_LANG to $PVT_LANG"
            python3 $ATTACK_DIR/google_translate.py \
                    --input_file $DATA_DIR/dataset/mc4/mc4.$ORG_LANG-100.jsonl \
                    --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra.jsonl \
                    --src_lang $ORG_LANG \
                    --tgt_lang $PVT_LANG \
                    --translation_part prompt

            echo "Generating with watermark using $WATERMARK_METHOD for $MODEL_NAME from $ORG_LANG to $PVT_LANG"
            # Generate with watermark
            python3 $WORK_DIR/gen.py \
                --base_model $MODEL_NAME \
                --fp16 \
                --batch_size $BATCH_SIZE \
                --input_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra.jsonl \
                --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.jsonl \
                $WATERMARK_METHOD_FLAG

            python3 $WORK_DIR/detect.py \
                --base_model $MODEL_NAME \
                --detect_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.jsonl \
                --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.z_score.jsonl \
                $WATERMARK_METHOD_FLAG

            # Apply CWRA attack
            python3 $ATTACK_DIR/google_translate.py \
                    --input_file  $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.jsonl \
                    --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.jsonl \
                    --src_lang $PVT_LANG \
                    --tgt_lang $ORG_LANG \
                    --translation_part response

            python3 $WORK_DIR/detect.py \
                --base_model $MODEL_NAME \
                --detect_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.jsonl \
                --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.z_score.jsonl \
                $WATERMARK_METHOD_FLAG

            if [ $WATERMARK_METHOD == "xsir" ]; then
                continue
            fi
            # Back translation
            python3 $ATTACK_DIR/google_translate.py \
                    --input_file  $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.jsonl \
                    --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra-back.mod.jsonl \
                    --src_lang $ORG_LANG \
                    --tgt_lang $PVT_LANG \
                    --translation_part response


            python3 $WORK_DIR/detect.py \
                --base_model $MODEL_NAME \
                --detect_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra-back.mod.jsonl \
                --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra-back.mod.z_score.jsonl \
                $WATERMARK_METHOD_FLAG
        done
    done
done