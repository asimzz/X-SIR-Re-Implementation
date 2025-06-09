set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
GEN_DIR=$WORK_DIR/gen


MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    # "bigscience/bloom-7b1"
    # "CohereForAI/aya-23-8B"
    # "facebook/xglm-564M"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    # "bloom-7b1"
    # "aya-23-8B"
    # "xglm-564M"
)

WATERMARK_METHODS=("xsir")
TGT_LANGS=(
    # High-resource languages
    # "fr"
    # "de"
    # "it"
    # "es"
    # "pt"
    # # Medium-resource languages
    # "pl"
    # "nl"
    # "ru"
    # "hi"
    # "ko"
    # "ja"
    # # Low-resource languages
    "bn"
    # "fa"
    # "vi"
    # "iw" # Hebrew
    # "uk"
    # "ta"
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
