# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Re-implementation of **X-SIR** (Cross-lingual watermark, [arXiv 2402.14007](https://arxiv.org/abs/2402.14007)) extended with **STEAM** — a detection-time, back-translation-based approach for multilingual watermarking — and a **per-text Bayesian Optimization** detector that searches over pivot languages instead of trying all of them ([arXiv 2510.18019](https://arxiv.org/abs/2510.18019)).

The upstream `README.md` documents the original X-SIR generation/detection commands. This file documents what's specific to this fork (STEAM, BO detector, γ-per-language normalization, the `scripts/` automation).

## Environment

- Python 3.10. `pip3 install -r requirements.txt`. A `.venv/` already exists in-tree.
- `.env` provides `OPENAI_API_KEY`, `HF_TOKEN`, `DEEPSEEK_API_KEY` — used by `attack/` translation/paraphrase scripts and gated HF model downloads.

## Pipeline at a glance

```
gen.py → detect.py → eval_detection.py                                (clean detection)
gen.py → attack/translate.py → detect.py → eval_detection.py          (translation attack)
back_eval_detection.py                                                (back-translation eval)
compute_gamma_lang.py → steam_bo_detector.py
                    └─→ evaluate_normalized_detection.py              (STEAM-BO evaluation)
```

All entry-point scripts accept `--help`.

| Script | Purpose |
| --- | --- |
| `gen.py` | Generate watermarked text from prompts (`mc4.{lang}.jsonl`). |
| `detect.py` | Compute raw z-scores for a `.jsonl` of texts. |
| `eval_detection.py` | AUC / TPR@FPR / F1 from paired `hum.z_score.jsonl` + `mod.z_score.jsonl`. |
| `back_eval_detection.py` | Evaluation under back-translation attack (target → pivot → target). |
| `compute_gamma_lang.py` | Pre-compute per-language empirical green-token fraction → `gamma_lang.json`. |
| `steam_bo_detector.py` | Per-text BO over pivot languages (the main contribution). |
| `evaluate_normalized_detection.py` | STEAM-aware metrics using γ-per-language-normalized z-scores. |

## STEAM-BO specifics (the non-obvious bits)

- **`compute_gamma_lang.py` MUST run before `steam_bo_detector.py`** — it produces `gamma_lang.json`, used to normalize z-scores per language and correct for tokenizer bias on low-resource languages.
- Normalization: `normalized_z = raw_z − mean_validation_z_per_language`. Across pivot candidates, STEAM picks the **max** normalized z-score.
- BO uses BoTorch `SingleTaskGP` + `LogExpectedImprovement` over a 131-D URIEL feature vector (`syntax_knn` 103-D + `phonology_knn` 28-D), assembled in [language_features.py](language_features.py).
- Defaults: 3 random initial pivots + up to 12 BO iterations (15 total evaluations). Tunable via `--n_initial`, `--max_evaluations`.
- Driver: `./run_steam_bo.sh [tgt_lang]`. With no arg, sweeps `(bn fa vi iw uk ta)`. Pass a single ISO 639-1 code to scope to one language.
- Validation z-scores are pre-computed in `mc4.{lang}.val.z_score.jsonl`, so detection-time back-translation only runs on the test set. The translator wrapper is [realtime_backtranslation.py](realtime_backtranslation.py) (Google via `deep_translator`, with caching).

## Watermark methods

Three implementations live under `src_watermark/`: [kgw/](src_watermark/kgw/), [xsir/](src_watermark/xsir/), [uw/](src_watermark/uw/). Selected via `--watermark_method {kgw|xsir|uw}` on both `gen.py` and `detect.py`.

- **KGW** — no extra flags. Standard config: `gamma=0.25, delta=2.0`, minhash seeding.
- **X-SIR / SIR** — additionally need `--transform_model data/model/transform_model_x-sbert_10K.pth`, `--embedding_model paraphrase-multilingual-mpnet-base-v2`, and `--mapping_file data/mapping/{xsir|sir}/300_mapping_{model_abbr}.json`. Standard config: `window=5, chunk=10, delta=1.0`.

## Models & languages

Models the user runs: `CohereForAI/aya-23-8B`, `meta-llama/Llama-3.2-1B`, `LLaMAX/LLaMAX3-8B`.

17 evaluation languages:
- High-resource: `fr de it es pt`
- Medium: `pl nl ru hi ko ja`
- Low: `bn fa vi iw uk ta`

Full set of translator-supported codes is in [supported_languages.txt](supported_languages.txt).

## Data layout

- **Prompts**: `data/dataset/mc4/mc4.{lang}.jsonl` (also `eli5/`, `multinews/`, `sts/`).
- **X-SIR artifacts**: `data/mapping/{xsir,sir}/`, `data/model/transform_model_x-sbert_10K.pth`, `data/embedding/`, `data/dictionary/`.
- **Outputs**: `gen/{model_abbr}/{method_seed}/`, with conventions:
  - `mc4.{lang}.mod.jsonl` — watermarked
  - `mc4.{lang}.hum.jsonl` — human baseline
  - `mc4.{lang}.val.jsonl` — validation split
  - `mc4.{src}-{tgt}.mod.jsonl` — post-translation
  - `*.z_score.jsonl` — detector output
  - `gamma_lang.json` — per-language empirical green fraction

## Helpers to reuse (don't reinvent)

- [language_code_converter.py](language_code_converter.py) — ISO 639-1 ↔ 639-3 (~90 langs); bridges translator and URIEL codes.
- [language_features.py](language_features.py) — combined URIEL syntax+phonology vectors via `lang2vec`.
- [realtime_backtranslation.py](realtime_backtranslation.py) — cached Google back-translation wrapper.
- [attack/call_openai.py](attack/call_openai.py), [attack/call_openrouter.py](attack/call_openrouter.py) — rate-limited LLM calls; RPM/TPM in [attack/const.py](attack/const.py).

## `scripts/` — batch drivers

Canonical way to run multi-model/multi-language sweeps. Don't write new ad-hoc shell scripts when one already exists. Grouped by purpose:

- Generation: `generate_with_watermark.sh`, `generate_with_watermark_translate.sh`, `generate_with_watermark_cwra.sh`, `generate_human_zscores.sh`, `generate_val.sh`, `generate_val_zscores.sh`
- Detection / evaluation: `detect.sh`, `eval.sh`, `eval_translate.sh`, `eval_back_translate.sh`, `eval_cwra.sh`
- STEAM prerequisites: `compute_gamma_lang.sh`, `compute_human_zscores.sh`
- X-SIR / SIR mapping training: `train_watermodel.sh`, `train_watermodel_10K.sh`, `xsir_mapping.sh`, `sir_mapping.sh`
- Misc: `setup.sh`, `gcp_ssh_connect.sh`, `playground*.sh`, `tools/`

## Smoke test (single language, end-to-end)

```bash
python3 gen.py    --base_model CohereForAI/aya-23-8B --watermark_method kgw \
                  --input_file  data/dataset/mc4/mc4.en.jsonl \
                  --output_file gen/aya-23-8B/kgw_seed0/mc4.en.mod.jsonl --fp16

python3 detect.py --base_model CohereForAI/aya-23-8B --watermark_method kgw \
                  --detect_file gen/aya-23-8B/kgw_seed0/mc4.en.mod.jsonl \
                  --output_file gen/aya-23-8B/kgw_seed0/mc4.en.mod.z_score.jsonl

python3 eval_detection.py \
  --hm_zscore gen/aya-23-8B/kgw_seed0/mc4.en.hum.z_score.jsonl \
  --wm_zscore gen/aya-23-8B/kgw_seed0/mc4.en.mod.z_score.jsonl
```

For STEAM-BO smoke: `./run_steam_bo.sh en` (requires `gen/aya-23-8B/kgw_seed0/gamma_lang.json` to exist).

## Git conventions

- Author: `git commit --author="asimzz <asimabdalla99@gmail.com>" ...`
- Format: `type: concise description` where type ∈ `{feat, fix, refactor, docs, test, chore, exp}`.
- Subject: imperative mood, lowercase, no period, ≤72 chars.
- Body explains **why**, not what. **Never** add `Co-Authored-By` lines.
