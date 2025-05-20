set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack

MAPPING_DIR=$DATA_DIR/mapping/xsir
TRANSFORM_MODEL=$DATA_DIR/model/transform_model_x-sbert.pth
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

BATCH_SIZE=32

MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "bigscience/bloom-7b1"
    "CohereForAI/aya-23-8B"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "bloom-7b1"
    "aya-23-8B"
)

WATERMARK_METHODS=("xsir")
TGT_LANGS=(
    "it"
    "es"
    "pt"
    "tr"
    "ar"
    "sw"
    "am"
    )
SEEDS=(0 42 123)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS must match"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            echo "Running $WATERMARK_METHOD with seed $SEED on $MODEL_NAME"

            MAPPING_FILE=$MAPPING_DIR/300_mapping_${MODEL_ABBR}_seed${SEED}.json

            if [ $WATERMARK_METHOD == "xsir" ]; then
                WATERMARK_FLAGS="--watermark_method xsir --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $MAPPING_FILE"
            else
                echo "Unknown watermark method: $WATERMARK_METHOD"
                exit 1
            fi

            OUT_DIR=$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${SEED}
            mkdir -p $OUT_DIR

            # 1. Generate with watermark
            python3 $WORK_DIR/gen.py \
                --base_model $MODEL_NAME \
                --fp16 \
                --batch_size $BATCH_SIZE \
                --input_file $DATA_DIR/dataset/mc4/mc4.en-100.jsonl \
                --output_file $OUT_DIR/mc4.en.mod.jsonl \
                $WATERMARK_FLAGS

            # 2. Detect watermark in English
            python3 $WORK_DIR/detect.py \
                --base_model $MODEL_NAME \
                --detect_file $OUT_DIR/mc4.en.mod.jsonl \
                --output_file $OUT_DIR/mc4.en.mod.z_score.jsonl \
                $WATERMARK_FLAGS

            # 3. Translate and detect for each target language
            for TGT_LANG in "${TGT_LANGS[@]}"; do
                # Translation Attack
                python3 $ATTACK_DIR/translate.py \
                    --input_file $OUT_DIR/mc4.en.mod.jsonl \
                    --output_file $OUT_DIR/mc4.en-${TGT_LANG}.mod.jsonl \
                    --model gpt-4o-mini \
                    --src_lang en \
                    --tgt_lang $TGT_LANG

                # Detect on Translated Output
                python3 $WORK_DIR/detect.py \
                    --base_model $MODEL_NAME \
                    --detect_file $OUT_DIR/mc4.en-${TGT_LANG}.mod.jsonl \
                    --output_file $OUT_DIR/mc4.en-${TGT_LANG}.mod.z_score.jsonl \
                    $WATERMARK_FLAGS
            done
        done
    done
done
