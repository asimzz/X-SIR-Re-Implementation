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
    # "meta-llama/Llama-2-7b-hf"
    # "baichuan-inc/Baichuan2-7B-Base"
    # "baichuan-inc/Baichuan-7B"
)

MODEL_ABBRS=(
     "bloom-7b1"
    "aya-23-8B"
    # "llama2-7b"
    # "baichuan2-7b"
    # "baichuan-7b"
)

WATERMARK_METHODS=(
    "kgw"
    # "xsir"
)

ORG_LANG="en"
PVT_LANGS=(
    "ar"
    "tr"
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

        for PVT_LANG in "${PVT_LANGS[@]}"; do
            echo "$MODEL_NAME $WATERMARK_METHOD (PVT_LANG: $PVT_LANG)"
            python3 $WORK_DIR/playground.py \
                --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
                --wm_no_cwra_zscore mc4.$ORG_LANG-$PVT_LANG-cwra.mod.z_score.jsonl \
                --wm_cwra_zscore mc4.$PVT_LANG-$ORG_LANG-cwra.mod.z_score.jsonl \
                --wm_cwra_back_zscore mc4.$ORG_LANG-$PVT_LANG-cwra-back.mod.z_score.jsonl \
                --cwra_attack_file cwra_attack-$PVT_LANG.txt \
                --no_cwra_attack_file no_cwra-$PVT_LANG.txt \
                --back_cwra_attack_file cwra_attack_back-$PVT_LANG.txt \
                --roc_curve $FIGURE_DIR/figure_$ORG_LANG-$PVT_LANG.pdf \
                --figure_title "$WATERMARK_METHOD (PVT_LANG: $PVT_LANG)"

            echo "======================================="
        done
    done
done