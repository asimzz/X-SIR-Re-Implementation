set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
GEN_DIR=$WORK_DIR/gen


MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

WATERMARK_METHODS=("kgw")
TGT_LANGS=(
    # High-resource languages
    "fr" # French
    "de" # German
    "it" # Italian
    "es" # Spanish
    "pt" # Portuguese
    # Medium-resource languages
    "pl" # Polish
    "nl" # Dutch
    "ru" # Russian
    "hi" # Hindi
    "ko" # Korean
    "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    "vi" # Vietnamese
    "iw" # Hebrew
    "uk" # Ukrainian
    "ta" # Tamil
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

            # echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) No-attack"
            # python3 $WORK_DIR/eval_detection.py \
            #     --hm_zscore $WATERMARK_DIR/mc4.en.hum.z_score.jsonl \
            #     --wm_zscore $WATERMARK_DIR/mc4.en.mod.z_score.jsonl

            echo "======================================="

            for TGT_LANG in "${TGT_LANGS[@]}"; do                
                # echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Translation ($TGT_LANG)"

                # python3 $WORK_DIR/eval_detection.py \
                #     --hm_zscore $WATERMARK_DIR/mc4.en-${TGT_LANG}.hum.z_score.jsonl \
                #     --wm_zscore $WATERMARK_DIR/mc4.en-${TGT_LANG}.mod.z_score.jsonl

                # echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Back-Translation without Normalization ($TGT_LANG)"
                # python3 $WORK_DIR/back_eval_detection.py \
                #     --tgt_lang "$TGT_LANG" \
                #     --base_wm_dir "$WATERMARK_DIR"

                echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Back-Translation with Normalization ($TGT_LANG)"
                python3 $WORK_DIR/evaluate_normalized_detection.py \
                    --tgt_lang "$TGT_LANG" \
                    --base_wm_dir "$WATERMARK_DIR"
            done
            echo "======================================="
        done
    done
done
