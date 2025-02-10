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

BATCH_SIZE=4

MODEL_NAMES=(
    "meta-llama/Llama-2-7b-hf"
    "baichuan-inc/Baichuan2-7B-Base"
    "baichuan-inc/Baichuan-7B"
)

MODEL_ABBRS=(
    "llama2-7b"
    "baichuan2-7b"
    "baichuan-7b"
)

WATERMARK_METHODS=(
    "kgw"
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
        if [ $WATERMARK_METHOD == "kgw" ]; then
            WATERMARK_METHOD_FLAG="--watermark_method kgw"
        else
            echo "Unknown watermark method: $WATERMARK_METHOD"
            exit 1
        fi

        echo "$MODEL_NAME $WATERMARK_METHOD Without CWRA Attack"
        python3 $WORK_DIR/eval_detection.py \
            --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
            --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-crwa.mod.z_score.jsonl

        echo "======================================="

        echo "$MODEL_NAME $WATERMARK_METHOD With CWRA Attack"
        python3 $WORK_DIR/eval_detection.py \
            --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
            --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$PVT_LANG-$ORG_LANG-crwa.mod.z_score.jsonl

        echo "======================================="

        echo "$MODEL_NAME $WATERMARK_METHOD With CWRA Attack (Back Translation)"
        python3 $WORK_DIR/eval_detection.py \
            --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
            --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-crwa-back.mod.z_score.jsonl

        echo "======================================="
    done
done