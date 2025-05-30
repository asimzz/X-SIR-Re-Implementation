set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
FIGURE_DIR=$WORK_DIR/data/figures
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

ORG_LANG="en"
PVT_LANGS=(
    "it"
    "es"
    "pt"
    "tr"
    "ar"
    "sw"
    "am"
)

ATTACK_TYPES=(
    "translation"
)

WATERMARK_METHOD="xsir"
SEEDS=(0 42 123)

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

                if [ $ATTACK_TYPE == "cwra" ]; then
                    ATTACK_TYPE_FLAG="--translation_file $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.z_score.jsonl"
                elif [ $ATTACK_TYPE == "translation" ]; then
                    ATTACK_TYPE_FLAG="--translation_file $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}/mc4.$ORG_LANG-$PVT_LANG.mod.z_score.jsonl"
                else
                    echo "Invalid attack type: $ATTACK_TYPE"
                    exit 1
                fi

                echo "Computing tokens distribution for $MODEL_NAME $WATERMARK_METHOD (seed=$SEED) $ATTACK_TYPE ($PVT_LANG)"
                python3 $WORK_DIR/src_watermark/$WATERMARK_METHOD/tokens_distribution.py \
                    --base_model $MODEL_ABBR \
                    --tgt_lang $PVT_LANG \
                    --figure_dir $FIGURE_DIR \
                    --seed $SEED \
                    $ATTACK_TYPE_FLAG
            done
        done
    done
done
