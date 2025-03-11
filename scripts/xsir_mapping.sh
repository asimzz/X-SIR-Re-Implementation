set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack

# Parameters for SIR/X-SIR
MAPPING_DIR=$WORK_DIR/data/mapping
TRANSFORM_MODEL=$WORK_DIR/data/model/transform_model_x-sbert.pth
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

BATCH_SIZE=16

MODEL_NAMES=(
    "bigscience/bloom-7b1"
    "CohereForAI/aya-23-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-2-7b-hf"
    "google/mt5-base"
    "facebook/m2m100_418M"
    "baichuan-inc/Baichuan2-7B-Base"
    "baichuan-inc/Baichuan-7B"
)

MODEL_ABBRS=(
    "bloom-7b1"
    "aya-23-8B"
    "llama-3.1-8B"
    "llama-3.2-1B"
    "llama2-7b"
    "mt5-base"
    "m2m100_418M"
    "baichuan2-7b"
    "baichuan-7b"
)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    echo "Generating semantic mappings for $MODEL_NAME"
    
    python3 $WORK_DIR/src_watermark/xsir/generate_semantic_mappings.py \
        --model $MODEL_NAME \
        --dictionary $DATA_DIR/dictionary/dictionary.txt \
        --output_file $DATA_DIR/mapping/xsir/300_mapping_$MODEL_ABBR.json
done