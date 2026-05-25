#!/usr/bin/env bash
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

# Load target languages from supported_languages.txt (one code per line).
# Lines that are empty or start with '#' are ignored.
SUPPORTED_LANGS_FILE="$WORK_DIR/supported_languages.txt"
if [ -f "$SUPPORTED_LANGS_FILE" ]; then
    TGT_LANGS=()
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|\#*) continue ;;
        esac
        TGT_LANGS+=("$line")
    done < "$SUPPORTED_LANGS_FILE"
else
    echo "supported languages file not found: $SUPPORTED_LANGS_FILE" >&2
    exit 1
fi
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
            CSV_FILE=$WATERMARK_DIR/detection_results.csv

            if [ ! -d "$WATERMARK_DIR" ]; then
                echo "Skipping missing directory: $WATERMARK_DIR" >&2
                continue
            fi

            echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) No-attack"
            python3 $WORK_DIR/eval_detection.py \
                --hm_zscore $WATERMARK_DIR/mc4.en.hum.z_score.jsonl \
                --wm_zscore $WATERMARK_DIR/mc4.en.mod.z_score.jsonl

            echo "======================================="

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                TRANS_HM=$WATERMARK_DIR/mc4.en-${TGT_LANG}.hum.z_score.jsonl
                TRANS_WM=$WATERMARK_DIR/mc4.en-${TGT_LANG}.mod.z_score.jsonl
                STEAM_HM=$WATERMARK_DIR/mc4.${TGT_LANG}.bo.hum.z_score.jsonl
                STEAM_WM=$WATERMARK_DIR/mc4.${TGT_LANG}.bo.z_score.jsonl

                if [ -f "$TRANS_HM" ] && [ -f "$TRANS_WM" ]; then
                    echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Translation ($TGT_LANG)"
                    python3 $WORK_DIR/eval_detection.py \
                        --hm_zscore "$TRANS_HM" \
                        --wm_zscore "$TRANS_WM" \
                        --language_code "$TGT_LANG" \
                        --csv_file "$CSV_FILE" \
                        --attack_type translation
                else
                    echo "Skipping translation for $TGT_LANG (missing files)" >&2
                fi

                if [ -f "$STEAM_HM" ] && [ -f "$STEAM_WM" ]; then
                    echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Translation STEAM ($TGT_LANG)"
                    python3 $WORK_DIR/eval_detection.py \
                        --hm_zscore "$STEAM_HM" \
                        --wm_zscore "$STEAM_WM" \
                        --language_code "$TGT_LANG" \
                        --csv_file "$CSV_FILE" \
                        --attack_type steam
                else
                    echo "Skipping STEAM for $TGT_LANG (missing files)" >&2
                fi
            done
            echo "CSV written to $CSV_FILE"
            echo "======================================="
        done
    done
done
