set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data


MAPPING_DIR=$WORK_DIR/data/mapping

MODEL_NAMES=(
    "meta-llama/Llama-2-7b-hf"
    "baichuan-inc/Baichuan2-7B-Base"
    "baichuan-inc/Baichuan-7B"
    "google/gemma-2b"
    "mistralai/Mistral-7B-v0.1"

)

MODEL_ABBRS=(
    "llama2-7b"
    "baichuan2-7b"
    "baichuan-7b"
    "gemma-2b"
    "mistral-7b"

)

SPLIT_TYPES=("random" "semantic")

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SPLIT_TYPE in "${SPLIT_TYPES[@]}"; do
        echo "Computing key space size for $MODEL_NAME using $SPLIT_TYPE split"

        python3  $WORK_DIR/key_space.py \
            --mapping_file $MAPPING_DIR/xsir/300_mapping_$MODEL_ABBR.json \
            --cluster_file $MAPPING_DIR/xsir/300_mapping_${MODEL_ABBR}_clusters.json \
            --split_type $SPLIT_TYPE

    done
done