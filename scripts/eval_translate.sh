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

MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "bigscience/bloom-7b1"
    "CohereForAI/aya-23-8B"
    "facebook/xglm-564M"
    "baichuan-inc/Baichuan2-7B-Base"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "bloom-7b1"
    "aya-23-8B"
    "xglm-564M"
    "baichuan2-7b"
)

WATERMARK_METHODS=(
    # "kgw"
    "xsir"
)

TGT_LANGS=(
    "it"
    "es"
    "pt"
    "de"
    "fr"
    "zh"
    "ja"
    "ar"
    "tr"
    "sw"
    "am"
)

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

        echo "$MODEL_NAME $WATERMARK_METHOD No-attack"
        python3 $WORK_DIR/eval_detection.py \
            --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
            --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.mod.z_score.jsonl

        echo "======================================="

        for TGT_LANG in "${TGT_LANGS[@]}"; do
            echo "$MODEL_NAME $WATERMARK_METHOD Translation ($TGT_LANG)"
            python3 $WORK_DIR/eval_detection.py \
                --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
                --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en-$TGT_LANG.mod.z_score.jsonl
            echo "Computing cluster tokens for $MODEL_NAME using $WATERMARK_METHOD for $TGT_LANG"
            python3 $WORK_DIR/src_watermark/xsir/cluster_tokens.py \
                --base_model $MODEL_NAME \
                --clusters_file $MAPPING_DIR/$WATERMARK_METHOD/300_mapping_${MODEL_ABBR}_clusters.json \
                --watermark_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.mod.jsonl \
                --translation_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en-$TGT_LANG.mod.jsonl
        done
        echo "======================================="
    done
done