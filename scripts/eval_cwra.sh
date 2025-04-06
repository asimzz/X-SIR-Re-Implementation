set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
FIGURE_DIR=$WORK_DIR/data/figures
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack
MAPPING_DIR=$WORK_DIR/data/mapping

MODEL_NAMES=(
    "bigscience/bloom-7b1"
    "facebook/xglm-564M"
    "CohereForAI/aya-23-8B"
    "meta-llama/Llama-3.2-1B"
    "baichuan-inc/Baichuan2-7B-Base"
)

MODEL_ABBRS=(
    "bloom-7b1"
    "xglm-564M"
    "aya-23-8B"
    "llama-3.2-1B"
    "baichuan2-7b"
)

WATERMARK_METHODS=(
    # "kgw"
    "xsir"
)

ORG_LANG="en"
PVT_LANGS=(
    "it"
    "es"
    "pt"
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
            echo "$MODEL_NAME $WATERMARK_METHOD Without CWRA Attack ($ORG_LANG -> $PVT_LANG)"
            python3 $WORK_DIR/eval_detection.py \
                --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
                --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.z_score.jsonl \
                --roc_curve $FIGURE_DIR/$MODEL_ABBR/$WATERMARK_METHOD/no_cwra-$PVT_LANG.txt

            echo "======================================="

            echo "$MODEL_NAME $WATERMARK_METHOD With CWRA Attack ($PVT_LANG -> $ORG_LANG)"
            python3 $WORK_DIR/eval_detection.py \
                --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
                --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.z_score.jsonl \
                --roc_curve $FIGURE_DIR/$MODEL_ABBR/$WATERMARK_METHOD/cwra_attack-$PVT_LANG.txt

            echo "======================================="

            if [ $WATERMARK_METHOD == "xsir" ]; then
                continue
            fi
            echo "$MODEL_NAME $WATERMARK_METHOD With CWRA Attack (Back Translation) ($ORG_LANG -> $PVT_LANG)"
            python3 $WORK_DIR/eval_detection.py \
                --hm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.en.hum.z_score.jsonl \
                --wm_zscore $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra-back.mod.z_score.jsonl \
                --roc_curve $FIGURE_DIR/$MODEL_ABBR/$WATERMARK_METHOD/cwra_attack_back-$PVT_LANG.txt

            echo "======================================="
        done
        for PVT_LANG in "${PVT_LANGS[@]}"; do
        echo "Computing cluster tokens for $MODEL_NAME using $WATERMARK_METHOD for $PVT_LANG"
            python3 $WORK_DIR/src_watermark/xsir/cluster_tokens.py \
                --base_model $MODEL_NAME \
                --clusters_file $MAPPING_DIR/$WATERMARK_METHOD/300_mapping_${MODEL_ABBR}_clusters.json \
                --input_file $GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.jsonl
        done
    done
done