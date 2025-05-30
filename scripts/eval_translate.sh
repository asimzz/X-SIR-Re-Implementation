set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack

MAPPING_DIR=$WORK_DIR/data/mapping/xsir
TRANSFORM_MODEL=$WORK_DIR/data/model/transform_model_x-sbert_10K.pth
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

BATCH_SIZE=32

MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "bigscience/bloom-7b1"
    "CohereForAI/aya-23-8B"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "bloom-7b1"
    "aya-23-8B"
)

WATERMARK_METHODS=("xsir")
TGT_LANGS=(
    "it"
    "es"
    "pt"
    "tr"
    "sw"
    "ar"
    "am"
    )
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

            WATERMARK_DIR=$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}
            MAPPING_FILE=$MAPPING_DIR/300_mapping_${MODEL_ABBR}_seed${SEED}.json
            CLUSTERS_FILE=${MAPPING_FILE/_mapping_/_mapping_}_clusters.json

            echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) No-attack"
            python3 $WORK_DIR/eval_detection.py \
                --hm_zscore $WATERMARK_DIR/mc4.en.hum.z_score.jsonl \
                --wm_zscore $WATERMARK_DIR/mc4.en.mod.z_score.jsonl

            echo "======================================="

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Translation ($TGT_LANG)"
                python3 $WORK_DIR/eval_detection.py \
                    --hm_zscore $WATERMARK_DIR/mc4.en.hum.z_score.jsonl \
                    --wm_zscore $WATERMARK_DIR/mc4.en-${TGT_LANG}.mod.z_score.jsonl
            done
            echo "======================================="
        done
    done
done
