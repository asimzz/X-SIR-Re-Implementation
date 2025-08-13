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
    "CohereForAI/aya-23-8B"
    "facebook/xglm-564M"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
    "xglm-564M"
)

ORG_LANG="en"
TGT_LANGS=(
    # High-resource languages
    "fr" # French
    # "de" # German
    # "it" # Italian
    # "es" # Spanish
    # "pt" # Portuguese
    # # Medium-resource languages
    # "pl" # Polish
    # "nl" # Dutch
    # "ru" # Russian
    # "hi" # Hindi
    # "ko" # Korean
    # "ja" # Japanese
    # # Low-resource languages
    # "bn" # Bengali
    # "fa" # Persian
    # "vi" # Vietnamese
    # "iw" # Hebrew
    # "uk" # Ukrainian
    # "ta" # Tamil
)

ORG_LANGS=(
    "en" # English
    # # High-resource languages
    # "fr" # French
    # "de" # German
    # "it" # Italian
    # "es" # Spanish
    # "pt" # Portuguese
    # # Medium-resource languages
    # "pl" # Polish
    # "nl" # Dutch
    # "ru" # Russian
    # "hi" # Hindi
    # "ko" # Korean
    # "ja" # Japanese
    # # Low-resource languages
    # "bn" # Bengali
    # "fa" # Persian
    # "vi" # Vietnamese
    # "iw" # Hebrew
    # "uk" # Ukrainian
    "ta" # Tamil
)

ATTACK_TYPES=(
    "translation"
)

WATERMARK_METHOD="kgw"
SEEDS=(0)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        for ORG_LANG in "${ORG_LANGS[@]}"; do
            for TGT_LANG in "${TGT_LANGS[@]}"; do
                for ATTACK_TYPE in "${ATTACK_TYPES[@]}"; do

                    if [ $ATTACK_TYPE == "cwra" ]; then
                        ATTACK_TYPE_FLAG="--translation_file $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}/mc4.$ORG_LANG-$TGT_LANG-cwra.mod.z_score.jsonl"
                    elif [ $ATTACK_TYPE == "translation" ]; then
                        ATTACK_TYPE_FLAG="--translation_file $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}/mc4.$TGT_LANG-$ORG_LANG-back.mod.z_score.jsonl"
                    else
                        echo "Invalid attack type: $ATTACK_TYPE"
                        exit 1
                    fi

                    echo "Computing tokens distribution for $MODEL_NAME $WATERMARK_METHOD (seed=$SEED) $ATTACK_TYPE ($ORG_LANG to $TGT_LANG)"
                    python3 $WORK_DIR/src_watermark/$WATERMARK_METHOD/tokens_distribution2.py \
                        --base_model $MODEL_ABBR \
                        --model_name $MODEL_NAME \
                        --tgt_lang $TGT_LANG \
                        --org_lang $ORG_LANG \
                        --figure_dir $FIGURE_DIR \
                        --seed $SEED \
                        $ATTACK_TYPE_FLAG
                done
            done
        done
    done
done
