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
WATERMARK_METHODS=("xsir")

TGT_LANGS=(
    "de"
    "fr"
    "zh"
    "ja"
)

ITER_NUM=({0..20})

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
        if [ $WATERMARK_METHOD == "kgw" ]; then
            WATERMARK_METHOD_FLAG="--watermark_method kgw"
        elif [ $WATERMARK_METHOD == "sir" ] || [ $WATERMARK_METHOD == "xsir" ]; then
            WATERMARK_METHOD_FLAG="--watermark_method xsir  --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_DIR/$WATERMARK_METHOD/300_mapping_$MODEL_ABBR.json"
        else
            echo "Unknown watermark method: $WATERMARK_METHOD"
            exit 1
        fi


        
        for ITER in "${ITER_NUM[@]}"; do
            echo "======================================="
            echo "Iteration $ITER"
            echo "$MODEL_NAME $WATERMARK_METHOD No-attack"
            python3 $WORK_DIR/eval_detection.py \
                --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/$ITER/mc4.en.hum.z_score.jsonl \
                --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/$ITER/mc4.en.mod.z_score.jsonl

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo "$MODEL_NAME $WATERMARK_METHOD Translation ($TGT_LANG)"
                python3 $WORK_DIR/eval_detection.py \
                    --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/$ITER/mc4.en-${TGT_LANG}.hum.z_score.jsonl \
                    --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/$ITER/mc4.en-$TGT_LANG.mod.z_score.jsonl
            done
        done

        echo "======================================="
    done
done