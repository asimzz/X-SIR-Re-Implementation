set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 $SCRIPT_DIR/build_dictionary.py \
    --dicts \
        $SCRIPT_DIR/download/en-de.txt \
        $SCRIPT_DIR/download/en-fr.txt \
        $SCRIPT_DIR/download/en-es.txt \
        $SCRIPT_DIR/download/en-it.txt \
        $SCRIPT_DIR/download/en-pt.txt \
        $SCRIPT_DIR/download/de-en.txt \
        $SCRIPT_DIR/download/de-fr.txt \
        $SCRIPT_DIR/download/de-es.txt \
        $SCRIPT_DIR/download/de-it.txt \
        $SCRIPT_DIR/download/de-pt.txt \
        $SCRIPT_DIR/download/fr-en.txt \
        $SCRIPT_DIR/download/fr-de.txt \
        $SCRIPT_DIR/download/fr-es.txt \
        $SCRIPT_DIR/download/fr-it.txt \
        $SCRIPT_DIR/download/fr-pt.txt \
        $SCRIPT_DIR/download/es-en.txt \
        $SCRIPT_DIR/download/es-de.txt \
        $SCRIPT_DIR/download/es-fr.txt \
        $SCRIPT_DIR/download/es-it.txt \
        $SCRIPT_DIR/download/es-pt.txt \
        $SCRIPT_DIR/download/it-en.txt \
        $SCRIPT_DIR/download/it-de.txt \
        $SCRIPT_DIR/download/it-fr.txt \
        $SCRIPT_DIR/download/it-es.txt \
        $SCRIPT_DIR/download/it-pt.txt \
        $SCRIPT_DIR/download/pt-en.txt \
        $SCRIPT_DIR/download/pt-de.txt \
        $SCRIPT_DIR/download/pt-fr.txt \
        $SCRIPT_DIR/download/pt-es.txt \
        $SCRIPT_DIR/download/pt-it.txt \
        $SCRIPT_DIR/download/en-pl.txt \
        $SCRIPT_DIR/download/en-nl.txt \
        $SCRIPT_DIR/download/en-ru.txt \
        $SCRIPT_DIR/download/en-hi.txt \
        $SCRIPT_DIR/download/en-ko.txt \
        $SCRIPT_DIR/download/en-ja.txt \
        $SCRIPT_DIR/download/pl-en.txt \
        $SCRIPT_DIR/download/nl-en.txt \
        $SCRIPT_DIR/download/ru-en.txt \
        $SCRIPT_DIR/download/hi-en.txt \
        $SCRIPT_DIR/download/ko-en.txt \
        $SCRIPT_DIR/download/ja-en.txt \
        $SCRIPT_DIR/download/en-bn.txt \
        $SCRIPT_DIR/download/en-fa.txt \
        $SCRIPT_DIR/download/en-vi.txt \
        $SCRIPT_DIR/download/en-he.txt \
        $SCRIPT_DIR/download/en-uk.txt \
        $SCRIPT_DIR/download/en-ta.txt \
        $SCRIPT_DIR/download/bn-en.txt \
        $SCRIPT_DIR/download/fa-en.txt \
        $SCRIPT_DIR/download/vi-en.txt \
        $SCRIPT_DIR/download/he-en.txt \
        $SCRIPT_DIR/download/uk-en.txt \
        $SCRIPT_DIR/download/ta-en.txt \
    --output_file $SCRIPT_DIR/dictionary.txt \
    # --add_meta_symbols