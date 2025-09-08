set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 $SCRIPT_DIR/build_dictionary.py \
    --dicts \
        $SCRIPT_DIR/download/fr-de.txt \
    --output_file $SCRIPT_DIR/dictionary-out-en.txt \
    # --add_meta_symbols