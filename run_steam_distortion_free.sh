#!/bin/bash
# STEAM evaluation for DISTORTION-FREE watermarks (ITS / EXP) with NON-ENGLISH source
# languages + random per-example attack. Mirrors run_steam_source_lang.sh (which is KGW), so
# ITS/EXP go through the SAME data, attack, STEAM defense, and eval — this is the evidence
# that STEAM is watermark-agnostic (rebuttal to the "only distortion-introducing watermarks"
# reviewer comment).
#
# ITS/EXP differences vs the KGW runner:
#   * No gamma/delta/gamma_lang.json. A one-time Stage 0 precomputes per-language null
#     distributions of the detection statistic (the ITS/EXP analogue of gamma_lang.json).
#   * gen.py uses a custom key-driven decode loop (--wm_key/--wm_n), batch_size 1.
#   * detect.py / steam_bo_detector.py score with -log(p_value) via the fast permutation test.
#
# Prerequisites:
#   * Source prompts data/dataset/mc4/mc4.{S}.jsonl and human negatives mc4.{S}.val.jsonl
#     (same as the KGW runner; run data/dataset/mc4/dl_data.py to mint missing sources).
#   * Cython installed (pip install -r requirements.txt) — the Levenshtein .pyx compiles on
#     first import via pyximport.
#
# Usage:
#   WATERMARK_METHOD=exp ./run_steam_distortion_free.sh                 # all sources, 500 texts
#   WATERMARK_METHOD=its ./run_steam_distortion_free.sh fr              # single source
#   NUM_TEXTS=5 MAX_EVALUATIONS=6 WATERMARK_METHOD=exp ./run_steam_distortion_free.sh fr  # smoke
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
WATERMARK_METHOD="${WATERMARK_METHOD:-exp}"    # its | exp
WM_KEY="${WM_KEY:-42}"
WM_N="${WM_N:-256}"
WM_K="${WM_K:-40}"                              # fixed alignment block (must match precompute)
WM_GAMMA="${WM_GAMMA:-1.0}"
WM_N_RUNS="${WM_N_RUNS:-100}"
BATCH_SIZE=1                                    # distortion-free decode loop: no left-pad artifacts
N_INITIAL=3
MAX_EVALUATIONS="${MAX_EVALUATIONS:-20}"
NUM_TEXTS="${NUM_TEXTS:-500}"
RANDOM_STATE=42

if [[ "$WATERMARK_METHOD" != "its" && "$WATERMARK_METHOD" != "exp" ]]; then
    echo "WATERMARK_METHOD must be 'its' or 'exp' (got '$WATERMARK_METHOD')"; exit 1
fi

OUT_DIR="$GEN_DIR/$MODEL_ABBR/${WATERMARK_METHOD}_seed${WM_KEY}"
NULL_DIR="$OUT_DIR/null"
mkdir -p "$OUT_DIR" "$NULL_DIR"

SRC_LANGS=("fr" "de" "es" "it" "pt" "ru" "ja")
if [[ $# -ge 1 ]]; then
    SRC_LANGS=("$1")
fi

WM_ARGS_GEN="--watermark_method $WATERMARK_METHOD --wm_key $WM_KEY --wm_n $WM_N"
WM_ARGS_DET="--watermark_method $WATERMARK_METHOD --wm_key $WM_KEY --wm_n $WM_N --wm_k $WM_K --wm_gamma $WM_GAMMA"

echo "🚀 STEAM distortion-free run | method=$WATERMARK_METHOD | model=$MODEL_ABBR | num_texts=$NUM_TEXTS | k=$WM_K"
echo "   sources: ${SRC_LANGS[*]}"

# ---- Stage 0: precompute per-language null distributions (candidate pivots) ----
# Nulls are needed for the pivot languages STEAM may back-translate into. We build them over
# the supported-language pool; this is the heavy one-time cost.
echo "🧮 [0] Precomputing null distributions -> $NULL_DIR ..."
mapfile -t PIVOT_LANGS < supported_languages.txt
python3 -m src_watermark.distortion_free.precompute_null \
    --base_model "$BASE_MODEL" --method "$WATERMARK_METHOD" \
    --wm_key "$WM_KEY" --wm_n "$WM_N" --wm_k "$WM_K" --wm_gamma "$WM_GAMMA" \
    --val_dir "$DATA_DIR/dataset/mc4" --output_dir "$NULL_DIR" \
    --langs "${PIVOT_LANGS[@]}" --num_texts "$NUM_TEXTS"

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
    echo "🧪 [1/5] Generating $WATERMARK_METHOD-watermarked text in $S ..."
    python3 gen.py \
        --base_model "$BASE_MODEL" \
        --fp16 \
        --batch_size "$BATCH_SIZE" \
        $WM_ARGS_GEN \
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

    # ---- Stage 3: no-defense baseline (detect directly on attacked text, fast path) ----
    # Detect in the source-language null (the text is a mix of attack langs; source S is the
    # natural reference for the "no back-translation" baseline).
    echo "📉 [3/5] No-defense baseline detection ..."
    python3 detect.py \
        --base_model "$BASE_MODEL" $WM_ARGS_DET --wm_fast --wm_null_file "$NULL_DIR/$S.npy" \
        --detect_file "$MOD_MIX" --output_file "$OUT_DIR/mc4.$S-mix.mod.z_score.jsonl"
    python3 detect.py \
        --base_model "$BASE_MODEL" $WM_ARGS_DET --wm_fast --wm_null_file "$NULL_DIR/$S.npy" \
        --detect_file "$HUM_MIX" --output_file "$OUT_DIR/mc4.$S-mix.hum.z_score.jsonl"

    # ---- Stage 4: STEAM per-example back-translation recovery ----
    echo "🔎 [4/5] STEAM per-example recovery detection ..."
    python3 steam_bo_detector.py \
        --per_example_attack \
        --watermark_method "$WATERMARK_METHOD" \
        --base_model "$BASE_MODEL" \
        --tgt_lang "$S" \
        --mod_file "$MOD_MIX" --hum_file "$HUM_MIX" \
        --input_dir "$OUT_DIR" --output_dir "$OUT_DIR" \
        --null_dir "$NULL_DIR" \
        --wm_key "$WM_KEY" --wm_n "$WM_N" --wm_k "$WM_K" --wm_gamma "$WM_GAMMA" \
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
    echo "📊 POOLED across sources: ${SRC_LANGS[*]} (method=$WATERMARK_METHOD)"
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
