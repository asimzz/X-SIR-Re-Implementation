#!/bin/bash

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"
ATTACK_DIR="$WORK_DIR/attack"
MAPPING_DIR="$DATA_DIR/mapping/xsir"

TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE=32

MODEL_NAMES=("meta-llama/Llama-3.2-1B")
MODEL_ABBRS=("llama-3.2-1B")

WATERMARK_METHODS=("xsir")
SEEDS=({0..49})
TGT_LANGS=("bn")  # example: Bengali, Hebrew

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "❌ MODEL_NAMES and MODEL_ABBRS length mismatch."
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for REF_SEED in "${SEEDS[@]}"; do
        echo -e "\n🧠 Using seed $REF_SEED as reference\n"

        for SEED in "${SEEDS[@]}"; do
            if [ "$SEED" -eq "$REF_SEED" ]; then
                echo "⏩ Skipping seed $SEED (same as reference)"
                continue
            fi

            for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
                echo "▶️ Running $WATERMARK_METHOD (seed=$SEED, ref_seed=$REF_SEED) on $MODEL_NAME"

                MAPPING_FILE="$MAPPING_DIR/$MODEL_ABBR/ref_seed$REF_SEED/300_mapping_${MODEL_ABBR}_seed${SEED}.json"
                OUT_DIR="$GEN_DIR/$MODEL_ABBR/$WATERMARK_METHOD/seed_${REF_SEED}"
                mkdir -p "$OUT_DIR"

                if [ "$WATERMARK_METHOD" == "xsir" ]; then
                    WATERMARK_FLAGS="--watermark_method xsir --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_FILE"
                else
                    echo "❌ Unknown watermark method: $WATERMARK_METHOD"
                    exit 1
                fi

                for TGT_LANG in "${TGT_LANGS[@]}"; do
                    echo "🌍 Translating and detecting for $TGT_LANG"

                    python3 "$WORK_DIR/detect.py" \
                        --base_model "$MODEL_NAME" \
                        --detect_file "$OUT_DIR/mc4.en-${TGT_LANG}.mod.jsonl" \
                        --output_file "$OUT_DIR/mc4.en-${TGT_LANG}-seed-${SEED}.mod.z_score.jsonl" \
                        $WATERMARK_FLAGS
                done
            done
        done
    done
done
