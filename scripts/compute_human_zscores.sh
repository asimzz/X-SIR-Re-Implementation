set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack

# Parameters for SIR/X-SIR
MAPPING_DIR=$WORK_DIR/data/mapping
TRANSFORM_MODEL=$WORK_DIR/data/model/transform_model_x-sbert.pth
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

# Model names and abbreviations
MODEL_NAMES=(
    # "ALLaM-AI/ALLaM-7B-Instruct-preview"
    # "QCRI/Fanar-1-9B-Instruct"
    # "Orbina/Orbita-v0.1"
    # "TURKCELL/Turkcell-LLM-7b-v1"
    "Finnish-NLP/Ahma-3B-Instruct"
)
MODEL_ABBRS=(
    # "allam-7b"
    # "fanar-1-9b"
    # "orbita-v0.1"
    # "turkcell-7b"
    "ahma-3b"
)

WATERMARK_METHODS=("xsir")
SEEDS=(0 42 123)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            echo "Generating detection for $MODEL_NAME using $WATERMARK_METHOD (seed=$SEED)"

            if [ $WATERMARK_METHOD == "kgw" ]; then
                WATERMARK_METHOD_FLAG="--watermark_method kgw"
            elif [ $WATERMARK_METHOD == "sir" ] || [ $WATERMARK_METHOD == "xsir" ]; then
                WATERMARK_METHOD_FLAG="--watermark_method xsir \
                    --transform_model $TRANSFORM_MODEL \
                    --embedding_model $EMBEDDING_MODEL \
                    --mapping_file $MAPPING_DIR/$WATERMARK_METHOD/300_mapping_${MODEL_ABBR}_seed${SEED}.json"
            else
                echo "Unknown watermark method: $WATERMARK_METHOD"
                exit 1
            fi

            python3 $WORK_DIR/detect.py \
                --base_model $MODEL_NAME \
                --detect_file $DATA_DIR/dataset/mc4/mc4.en.jsonl \
                --output_file $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}/mc4.en.hum.z_score.jsonl \
                $WATERMARK_METHOD_FLAG
        done
    done
done
