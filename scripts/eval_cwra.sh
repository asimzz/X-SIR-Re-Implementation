set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
FIGURE_DIR=$WORK_DIR/data/figures
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack
MAPPING_DIR=$WORK_DIR/data/mapping

# Model names and abbreviations
MODEL_NAMES=(
    # "ALLaM-AI/ALLaM-7B-Instruct-preview"
    # "QCRI/Fanar-1-9B-Instruct"
    # "Orbina/Orbita-v0.1"
    # "TURKCELL/Turkcell-LLM-7b-v1"
    "Finnish-NLP/Ahma-3B-Instruct"
)
MODEL_ABBRS=(
    # "allam-7b"
    # "fanar-1-9b"
    # "orbita-v0.1"
    # "turkcell-7b"
    "ahma-3b"
)
# Settings
WATERMARK_METHODS=("xsir")
SEEDS=(0 42 123)
ORG_LANG="en"
PVT_LANGS=("fi")

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
                echo "$MODEL_NAME $WATERMARK_METHOD Without CWRA Attack ($ORG_LANG -> $PVT_LANG)"
                python3 $WORK_DIR/eval_detection.py \
                    --hm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.en.hum.z_score.jsonl \
                    --wm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.$ORG_LANG-$PVT_LANG-cwra.mod.z_score.jsonl \
                    --roc_curve $FIGURE_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/no_cwra-$PVT_LANG.txt

                echo "======================================="

                echo "$MODEL_NAME $WATERMARK_METHOD With CWRA Attack ($PVT_LANG -> $ORG_LANG)"
                python3 $WORK_DIR/eval_detection.py \
                    --hm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.en.hum.z_score.jsonl \
                    --wm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.$PVT_LANG-$ORG_LANG-cwra.mod.z_score.jsonl \
                    --roc_curve $FIGURE_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/cwra_attack-$PVT_LANG.txt

                echo "======================================="

                if [ $WATERMARK_METHOD == "xsir" ]; then
                    continue
                fi
                echo "$MODEL_NAME $WATERMARK_METHOD With CWRA Attack (Back Translation) ($ORG_LANG -> $PVT_LANG)"
                python3 $WORK_DIR/eval_detection.py \
                    --hm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.en.hum.z_score.jsonl \
                    --wm_zscore $GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/mc4.$ORG_LANG-$PVT_LANG-cwra-back.mod.z_score.jsonl \
                    --roc_curve $FIGURE_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed$SEED/cwra_attack_back-$PVT_LANG.txt

                echo "======================================="
            done
        done
    done
done