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
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-2-7b-hf"
    "google/mt5-base"
    "facebook/m2m100_418M"
    "baichuan-inc/Baichuan2-7B-Base"
    "baichuan-inc/Baichuan-7B"
)

MODEL_ABBRS=(
    "bloom-7b1"
    "aya-23-8B"
    "llama-3.1-8B"
    "llama-3.2-1B"
    "llama2-7b"
    "mt5-base"
    "m2m100_418M"
    "baichuan2-7b"
    "baichuan-7b"
)

WATERMARK_METHODS=(
    "kgw"
    "xsir"
)

ORG_LANG="en"
PVT_LANG="zh"

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
        else
            echo "Unknown watermark method: $WATERMARK_METHOD"
            exit 1
        fi

        python3 $WORK_DIR/detect.py \
            --base_model $MODEL_NAME \
            --detect_file $DATA_DIR/dataset/mc4/mc4.$ORG_LANG-100.jsonl \
            --output_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG.hum.z_score.jsonl \
            $WATERMARK_METHOD_FLAG

    done
done