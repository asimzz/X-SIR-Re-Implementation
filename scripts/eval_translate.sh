set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
GEN_DIR=$WORK_DIR/gen

# Model names and abbreviations
MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)
MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

# Settings
WATERMARK_METHODS=("xsir")
SEEDS=(0 42 123)
TGT_LANGS=(
    "en" # English
    "fr" # French
    "de" # German
    "zh" # Chinese
    "ja" # Japanese
)

ORG_LANGS=("en" "fr" "de" "zh")

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for TGT_LANG in "${TGT_LANGS[@]}"; do

        for SEED in "${SEEDS[@]}"; do
            for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do

                WATERMARK_DIR=$GEN_DIR/$MODEL_ABBR/${TGT_LANG}/${WATERMARK_METHOD}_seed${SEED}

                for ORG_LANG in "${ORG_LANGS[@]}"; do
                    if [ "$ORG_LANG" == "$TGT_LANG" ]; then
                        continue
                    fi

                    echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) No-attack ($ORG_LANG)"
                    python3 $WORK_DIR/eval_detection.py \
                        --hm_zscore $WATERMARK_DIR/mc4.${ORG_LANG}.hum.z_score.jsonl \
                        --wm_zscore $WATERMARK_DIR/mc4.${ORG_LANG}.mod.z_score.jsonl

                        echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Translation from (${ORG_LANG}) to ($TGT_LANG)"
                        python3 $WORK_DIR/eval_detection.py \
                            --hm_zscore $WATERMARK_DIR/mc4.$ORG_LANG-${TGT_LANG}.hum.z_score.jsonl \
                            --wm_zscore $WATERMARK_DIR/mc4.$ORG_LANG-${TGT_LANG}.mod.z_score.jsonl
                done
                echo "======================================="
            done
        done
    done
done
