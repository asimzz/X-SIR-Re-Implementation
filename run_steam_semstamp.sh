#!/bin/bash
# STEAM evaluation for the SEMSTAMP semantic watermark with NON-ENGLISH source languages +
# random per-example attack. Mirrors run_steam_source_lang.sh (KGW) so SemStamp runs through
# the SAME data, attack, STEAM defense, and eval — more evidence that STEAM is watermark
# agnostic (rebuttal to the "distortion-free / semantic schemes" reviewer comment).
#
# SemStamp specifics:
#   * Sentence-level LSH watermark via rejection sampling (gen.py --watermark_method semstamp,
#     batch_size 1). Uses a MULTILINGUAL sentence encoder so source + pivot languages embed
#     consistently.
#   * Detection = sentence-level KGW-style z-score with green/total SENTENCE counts, so STEAM
#     reuses the KGW gamma_lang scoring path. Stage 0 computes that per-language gamma_lang
#     (sentence-level) via compute_gamma_lang_semstamp.py.
#
# Prerequisites: source prompts mc4.{S}.jsonl + human negatives mc4.{S}.val.jsonl (as KGW),
#   sentence-transformers + nltk punkt installed (pip install -r requirements.txt).
#
# Usage:
#   ./run_steam_semstamp.sh                 # all sources, 500 texts
#   ./run_steam_semstamp.sh fr              # single source
#   NUM_TEXTS=5 MAX_EVALUATIONS=6 ./run_steam_semstamp.sh fr   # smoke test
set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR"
cd "$WORK_DIR"   # steam_bo_detector.py reads supported_languages.txt from cwd
DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/gen"

# Configuration
BASE_MODEL="CohereForAI/aya-23-8B"
MODEL_ABBR="aya-23-8B"
EMBEDDER="${EMBEDDER:-paraphrase-multilingual-mpnet-base-v2}"
SP_DIM="${SP_DIM:-3}"
LMBD="${LMBD:-0.25}"
MARGIN="${MARGIN:-0.0}"
BATCH_SIZE=1                                # SemStamp generates one prompt at a time
N_INITIAL=3
MAX_EVALUATIONS="${MAX_EVALUATIONS:-20}"
NUM_TEXTS="${NUM_TEXTS:-500}"
RANDOM_STATE=42

OUT_DIR="$GEN_DIR/$MODEL_ABBR/semstamp_dim${SP_DIM}"
GAMMA_LANG_FILE="$OUT_DIR/gamma_lang.json"
mkdir -p "$OUT_DIR"

WM_GEN="--watermark_method semstamp --embedding_model $EMBEDDER --sp_dim $SP_DIM --lmbd $LMBD --margin $MARGIN"
WM_DET="--watermark_method semstamp --embedding_model $EMBEDDER --sp_dim $SP_DIM --lmbd $LMBD"

SRC_LANGS=("fr" "de" "es" "it" "pt" "ru" "ja")
if [[ $# -ge 1 ]]; then
    SRC_LANGS=("$1")
fi

echo "🚀 STEAM SemStamp run | model=$MODEL_ABBR | embedder=$EMBEDDER | sp_dim=$SP_DIM | num_texts=$NUM_TEXTS"
echo "   sources: ${SRC_LANGS[*]}"

# ---- Stage 0: sentence-level gamma_lang over the candidate pivot pool ----
echo "🧮 [0] Computing sentence-level gamma_lang -> $GAMMA_LANG_FILE ..."
mapfile -t PIVOT_LANGS < supported_languages.txt
python3 compute_gamma_lang_semstamp.py \
    --input_dir "$DATA_DIR/dataset/mc4" --output_file "$GAMMA_LANG_FILE" \
    --embedding_model "$EMBEDDER" --sp_dim "$SP_DIM" --lmbd "$LMBD" \
    --num_texts "$NUM_TEXTS" --langs "${PIVOT_LANGS[@]}"

for S in "${SRC_LANGS[@]}"; do
    echo ""
    echo "=================================================="
    echo "🎯 Source language: $S"
    echo "=================================================="

    SRC_PROMPT="$DATA_DIR/dataset/mc4/mc4.$S.jsonl"
    SRC_VAL="$DATA_DIR/dataset/mc4/mc4.$S.val.jsonl"
    if [ ! -f "$SRC_PROMPT" ]; then
        echo "⚠️  Missing prompts $SRC_PROMPT (run dl_data.py). Skipping $S."
        continue
    fi
    if [ ! -f "$SRC_VAL" ]; then
        echo "⚠️  Missing human negatives $SRC_VAL. Skipping $S."
        continue
    fi

    MOD_RAW="$OUT_DIR/mc4.$S.mod.jsonl"
    HUM_RAW="$OUT_DIR/mc4.$S.hum.jsonl"

    GEN_IN="$SRC_PROMPT"
    if [ "$NUM_TEXTS" -lt 500 ]; then
        GEN_IN="$OUT_DIR/mc4.$S.prompt.head${NUM_TEXTS}.jsonl"
        head -n "$NUM_TEXTS" "$SRC_PROMPT" > "$GEN_IN"
        head -n "$NUM_TEXTS" "$SRC_VAL" > "$HUM_RAW"
    else
        cp "$SRC_VAL" "$HUM_RAW"
    fi

    # ---- Stage 1: generate native watermarked positives in S ----
    echo "🧪 [1/5] Generating SemStamp-watermarked text in $S ..."
    python3 gen.py \
        --base_model "$BASE_MODEL" \
        --fp16 \
        --batch_size "$BATCH_SIZE" \
        $WM_GEN \
        --input_file "$GEN_IN" \
        --output_file "$MOD_RAW"

    # ---- Stage 2: random per-example attack (same seed for mod & hum) ----
    MOD_MIX="$OUT_DIR/mc4.$S-mix.mod.jsonl"
    HUM_MIX="$OUT_DIR/mc4.$S-mix.hum.jsonl"
    echo "🌍 [2/5] Random per-example attack (mod & hum) ..."
    python3 attack/random_attack_translate.py \
        --input_file "$MOD_RAW" --output_file "$MOD_MIX" \
        --src_lang "$S" --translation_part response --random_state "$RANDOM_STATE"
    python3 attack/random_attack_translate.py \
        --input_file "$HUM_RAW" --output_file "$HUM_MIX" \
        --src_lang "$S" --translation_part response --random_state "$RANDOM_STATE"

    # ---- Stage 3: no-defense baseline (detect directly on attacked text) ----
    echo "📉 [3/5] No-defense baseline detection ..."
    python3 detect.py \
        --base_model "$BASE_MODEL" $WM_DET \
        --detect_file "$MOD_MIX" --output_file "$OUT_DIR/mc4.$S-mix.mod.z_score.jsonl"
    python3 detect.py \
        --base_model "$BASE_MODEL" $WM_DET \
        --detect_file "$HUM_MIX" --output_file "$OUT_DIR/mc4.$S-mix.hum.z_score.jsonl"

    # ---- Stage 4: STEAM per-example back-translation recovery ----
    echo "🔎 [4/5] STEAM per-example recovery detection ..."
    python3 steam_bo_detector.py \
        --per_example_attack \
        --watermark_method semstamp \
        --base_model "$BASE_MODEL" \
        --tgt_lang "$S" \
        --mod_file "$MOD_MIX" --hum_file "$HUM_MIX" \
        --input_dir "$OUT_DIR" --output_dir "$OUT_DIR" \
        --gamma_lang_file "$GAMMA_LANG_FILE" \
        --embedding_model "$EMBEDDER" --sp_dim "$SP_DIM" --lmbd "$LMBD" \
        --n_initial "$N_INITIAL" --max_evaluations "$MAX_EVALUATIONS" \
        --num_texts "$NUM_TEXTS" --random_state "$RANDOM_STATE"

    # ---- Stage 5a: per-source metrics ----
    echo "📈 [5/5] Per-source metrics for $S:"
    echo "  -- STEAM --"
    python3 eval_detection.py \
        --wm_zscore "$OUT_DIR/mc4.$S-mix.bo.z_score.jsonl" \
        --hm_zscore "$OUT_DIR/mc4.$S-mix.bo.hum.z_score.jsonl"
    echo "  -- No-defense baseline --"
    python3 eval_detection.py \
        --wm_zscore "$OUT_DIR/mc4.$S-mix.mod.z_score.jsonl" \
        --hm_zscore "$OUT_DIR/mc4.$S-mix.hum.z_score.jsonl"
done

# ---- Stage 5b: pooled + per-tier across all processed sources ----
STEAM_WM=(); STEAM_HM=(); BASE_WM=(); BASE_HM=()
for S in "${SRC_LANGS[@]}"; do
    f="$OUT_DIR/mc4.$S-mix.bo.z_score.jsonl"
    [ -f "$f" ] || continue
    STEAM_WM+=("$OUT_DIR/mc4.$S-mix.bo.z_score.jsonl")
    STEAM_HM+=("$OUT_DIR/mc4.$S-mix.bo.hum.z_score.jsonl")
    BASE_WM+=("$OUT_DIR/mc4.$S-mix.mod.z_score.jsonl")
    BASE_HM+=("$OUT_DIR/mc4.$S-mix.hum.z_score.jsonl")
done

if [ "${#STEAM_WM[@]}" -gt 0 ]; then
    echo ""
    echo "=================================================="
    echo "📊 POOLED across sources: ${SRC_LANGS[*]} (semstamp)"
    echo "=================================================="
    cat "${STEAM_WM[@]}" > "$OUT_DIR/pooled.bo.z_score.jsonl"
    cat "${STEAM_HM[@]}" > "$OUT_DIR/pooled.bo.hum.z_score.jsonl"
    cat "${BASE_WM[@]}"  > "$OUT_DIR/pooled.base.mod.z_score.jsonl"
    cat "${BASE_HM[@]}"  > "$OUT_DIR/pooled.base.hum.z_score.jsonl"

    echo "  -- POOLED STEAM --"
    python3 eval_detection.py \
        --wm_zscore "$OUT_DIR/pooled.bo.z_score.jsonl" \
        --hm_zscore "$OUT_DIR/pooled.bo.hum.z_score.jsonl"
    echo "  -- POOLED baseline --"
    python3 eval_detection.py \
        --wm_zscore "$OUT_DIR/pooled.base.mod.z_score.jsonl" \
        --hm_zscore "$OUT_DIR/pooled.base.hum.z_score.jsonl"

    echo ""
    echo "📊 Per attack-language tier (STEAM vs baseline):"
    python3 evaluate_by_tier.py \
        --steam_wm "${STEAM_WM[@]}" --steam_hm "${STEAM_HM[@]}" \
        --baseline_wm "${BASE_WM[@]}" --baseline_hm "${BASE_HM[@]}" \
        --output_json "$OUT_DIR/by_tier_summary.json"
fi

echo ""
echo "✅ Done. Outputs in: $OUT_DIR"
