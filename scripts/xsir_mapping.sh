set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack

# Parameters for SIR/X-SIR
MAPPING_DIR=$DATA_DIR/mapping
TRANSFORM_MODEL=$DATA_DIR/model/transform_model_x-sbert.pth
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
    "LLaMAX/LLaMAX3-8B"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
    "llamax3-8B"
)


# Languages for which to create mappings without it
OUT_LANGS=("en" "de" "fr" "ja" "zh")

SEEDS=(0 42 123)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for LANG in "${OUT_LANGS[@]}"; do

    echo "Generating mappings without language: $LANG"
    for i in "${!MODEL_NAMES[@]}"; do
        MODEL_NAME=${MODEL_NAMES[$i]}
        MODEL_ABBR=${MODEL_ABBRS[$i]}

        for SEED in "${SEEDS[@]}"; do
            echo "Generating semantic mappings for $MODEL_NAME with seed $SEED"
            
            python3 $WORK_DIR/src_watermark/xsir/generate_semantic_mappings.py \
                --model "$MODEL_NAME" \
                --dictionary "$DATA_DIR/dictionary/dictionary-out-$LANG.txt" \
                --output_file "$MAPPING_DIR/xsir/${LANG}/300_mapping_${MODEL_ABBR}_seed${SEED}.json" \
                --seed "$SEED"
        done
    done
done
