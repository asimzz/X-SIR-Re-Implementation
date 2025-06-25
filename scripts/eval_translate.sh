set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
GEN_DIR=$WORK_DIR/gen



# Model names and abbreviations
MODEL_NAMES=(
    "ALLaM-AI/ALLaM-7B-Instruct-preview"
    # "QCRI/Fanar-1-9B-Instruct"
    # "Orbina/Orbita-v0.1"
    # "TURKCELL/Turkcell-LLM-7b-v1"
    # "Finnish-NLP/Ahma-3B-Instruct"
)
MODEL_ABBRS=(
    "allam-7b"
    # "fanar-1-9b"
    # "orbita-v0.1"
    # "turkcell-7b"
    # "ahma-3b"
)

WATERMARK_METHODS=("xsir")
TGT_LANGS=("ar")
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
