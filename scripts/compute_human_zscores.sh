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

MODEL_NAMES=(
    "facebook/xglm-564M"
    "mistralai/Mistral-Small-24B-Instruct-2501"
    "utter-project/EuroLLM-9B-Instruct"

)

MODEL_ABBRS=(
    "xglm-564M"
    "Mistral-Small-24B-Instruct-2501"
    "EuroLLM-9B-Instruct"
)

WATERMARK_METHODS=(
    # "kgw"
    "xsir"
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

        python3 $WORK_DIR/detect.py \
                --base_model $MODEL_NAME \
                --detect_file $DATA_DIR/dataset/mc4/mc4.en-100.jsonl \
                --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
                $WATERMARK_METHOD_FLAG
    done
done