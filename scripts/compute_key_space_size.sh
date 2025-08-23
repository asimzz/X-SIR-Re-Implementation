set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data


MAPPING_DIR=$WORK_DIR/data/mapping

# Model names and abbreviations
MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
    "google/gemma-3-4b-pt"
)
MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
    "gemma-3-4b-pt"
)

SPLIT_TYPES=("cluster" "semantic")

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
            --base_model $MODEL_NAME \
            --cluster_file $MAPPING_DIR/xsir/300_mapping_${MODEL_ABBR}_seed0_clusters.json \
            --split_type $SPLIT_TYPE

    done
done