set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
FIGURE_DIR=$WORK_DIR/data/figures
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack
MAPPING_DIR=$WORK_DIR/data/mapping

# Model configurations
MODEL_NAMES=(
    # "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
    # "LLaMAX/LLaMAX3-8B"
)

MODEL_ABBRS=(
    # "llama-3.2-1B"
    "aya-23-8B"
    # "llamax3-8B"
)

# Settings
WATERMARK_METHODS=("xsir")
SEEDS=(0 42 123)
ORG_LANG="en"
PVT_LANGS=("fr" "de" "zh" "ja")

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do

        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            echo "▶️ Evaluating $WATERMARK_METHOD (seed=$SEED) on $MODEL_NAME"
            for PVT_LANG in "${PVT_LANGS[@]}"; do
                # echo "$MODEL_NAME $WATERMARK_METHOD Without CWRA Attack ($ORG_LANG -> $PVT_LANG)"
                # python3 $WORK_DIR/eval_detection.py \
                #     --hm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.en.hum.z_score.jsonl \
                #     --wm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.z_score.jsonl

                # echo "======================================="

                echo "$MODEL_NAME $WATERMARK_METHOD With CWRA Attack ($PVT_LANG -> $ORG_LANG)"
                python3 $WORK_DIR/eval_detection.py \
                    --hm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.en.hum.z_score.jsonl \
                    --wm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.z_score.jsonl
            done
        done
    done
done