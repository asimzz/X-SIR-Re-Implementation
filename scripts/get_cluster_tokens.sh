set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
MAPPING_DIR=$WORK_DIR/data/mapping

BATCH_SIZE=8

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

SEEDS=("0" "42" "123")

ORG_LANG="en"
PVT_LANGS=(
    "it"
    "es"
    "pt"
    "ar"
    "tr"
    "sw"
    "am"
)

ATTACK_TYPES=(
    "translation"
    # "cwra"
)

WATERMARK_METHOD="xsir"

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        for PVT_LANG in "${PVT_LANGS[@]}"; do
            for ATTACK_TYPE in "${ATTACK_TYPES[@]}"; do
                GEN_SEED_DIR=$GEN_DIR/$MODEL_ABBR/xsir_seed$SEED

                if [ $ATTACK_TYPE == "cwra" ]; then
                    ATTACK_TYPE_FLAG="--watermark_file $GEN_SEED_DIR/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.jsonl --translation_file $GEN_SEED_DIR/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.jsonl"
                elif [ $ATTACK_TYPE == "translation" ]; then
                    ATTACK_TYPE_FLAG="--watermark_file $GEN_SEED_DIR/mc4.$ORG_LANG.mod.jsonl --translation_file $GEN_SEED_DIR/mc4.$ORG_LANG-$PVT_LANG.mod.jsonl"
                else
                    echo "Invalid attack type: $ATTACK_TYPE"
                    exit 1
                fi

                echo "Computing cluster tokens for $MODEL_NAME | $WATERMARK_METHOD | seed $SEED | lang $PVT_LANG"

                python3 $WORK_DIR/src_watermark/$WATERMARK_METHOD/cluster_tokens.py \
                    --base_model "$MODEL_NAME" \
                    --clusters_file "$MAPPING_DIR/$WATERMARK_METHOD/300_mapping_${MODEL_ABBR}_clusters.json" \
                    $ATTACK_TYPE_FLAG
            done
        done
    done
done
