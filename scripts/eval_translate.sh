set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
GEN_DIR=$WORK_DIR/gen


MODEL_NAMES=(
    "CohereForAI/aya-23-8B"
)

MODEL_ABBRS=(
    "aya-23-8B"
)

WATERMARK_METHODS=("kgw")

PVT_LANGS=(
    "de"
    "ko"
    "bn"
)
TGT_LANGS=(
    # # High-resource languages
    "fr"
    "de"
    "it"
    "es"
    "pt"
    # Medium-resource languages
    "pl"
    "nl"
    "ru"
    "hi"
    "ko"
    "ja"
    # Low-resource languages
    "bn"
    "fa"
    "vi"
    "iw" # Hebrew
    "uk"
    "ta"
    )
SEEDS=(0)

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

            echo "======================================="

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                for PVT_LANG in "${PVT_LANGS[@]}"; do
                    if [ "$TGT_LANG" == "$PVT_LANG" ]; then
                        continue
                    fi

                    echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Multi-step Translation Attack  ($TGT_LANG)-> ($PVT_LANG)"
                    python3 $WORK_DIR/eval_detection.py \
                        --hm_zscore $WATERMARK_DIR/mc4.${TGT_LANG}-${PVT_LANG}-pivot.bo.hum.z_score.jsonl \
                        --wm_zscore $WATERMARK_DIR/mc4.${TGT_LANG}-${PVT_LANG}-pivot.bo.z_score.jsonl
                done
               
            done
            echo "======================================="
        done
    done
done
