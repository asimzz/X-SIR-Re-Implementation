set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
FIGURE_DIR=$WORK_DIR/data/figures
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack

MODEL_NAMES=(
    "bigscience/bloom-7b1"
    "CohereForAI/aya-23-8B"
    "meta-llama/Llama-3.2-1B"
    "facebook/xglm-564M"
    "baichuan-inc/Baichuan2-7B-Base"
)

MODEL_ABBRS=(
    "bloom-7b1"
    "aya-23-8B"
    "llama-3.2-1B"
    "xglm-564M"
    "baichuan2-7b"
)

WATERMARK_METHODS=(
    # "kgw"
    # "sir"
    "xsir"
)

ORG_LANG="en"
TGT_LANGS=(
    "it"
    "es"
    "de"
    "fr"
    "zh"
    "ja"
)


if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do

        for TGT_LANG in "${TGT_LANGS[@]}"; do
            echo "$MODEL_NAME $WATERMARK_METHOD (TGT_LANG: $TGT_LANG)"
            python3 $WORK_DIR/playground_translate.py \
                --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
                --wm_no_translation_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG.mod.z_score.jsonl \
                --wm_translation_attack_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$TGT_LANG.mod.z_score.jsonl \
                --no_translation_attack_file no_translation-$TGT_LANG.txt \
                --translation_attack_file translation_attack-$TGT_LANG.txt \
                --roc_curve $FIGURE_DIR/$MODEL_ABBR/figure_translation_$ORG_LANG-$TGT_LANG.png \
                --model_abbr $MODEL_ABBR \
                --tgt_lang $TGT_LANG \
                --figure_title "$WATERMARK_METHOD Translation Attack (TGT_LANG: $TGT_LANG)"

            echo "======================================="
        done
    done
done