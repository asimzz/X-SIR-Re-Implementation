import os
import json
import jieba
from opencc import OpenCC
from datasets import load_dataset
from tqdm import tqdm

T2S = OpenCC('t2s')
N = 500

# Non-English source languages for the "generated language != English" experiment.
# en/zh/de/fr already exist and are skipped below; add the new source languages here.
# (es, it, pt, ru high/medium-resource; ja medium.)
LANGS = ["es", "it", "pt", "ru", "ja"]


def load_mc4_stream(lang):
    """Stream the mC4 validation split for `lang`.

    The legacy HF `mc4` config is gated/deprecated on some setups; fall back to
    `allenai/c4` (same schema, `text` field) if it is unavailable.
    """
    try:
        return load_dataset("mc4", lang, streaming=True, split="validation")
    except Exception as e:
        print(f"⚠️ load_dataset('mc4', '{lang}') failed ({e}); falling back to allenai/c4")
        return load_dataset("allenai/c4", lang, streaming=True, split="validation")


for lang in LANGS:
    out_path = f"mc4.{lang}.jsonl"
    if os.path.exists(out_path):
        print(f"✅ {out_path} already exists, skipping")
        continue
    bar = tqdm(total=N, desc=f"Processing {lang}")
    ds = load_mc4_stream(lang)
    prompts = []
    responses = []
    for s in ds:
        text = s["text"]
        if lang == "zh":
            text = T2S.convert(text)
        tokens = jieba.cut(text)
        tokens = list(tokens)
        if 195 <= len(tokens) <= 205:
            split_index = int(len(tokens) * 0.1)

            front_tokens = tokens[:split_index]
            back_tokens = tokens[split_index:]

            prompt = "".join(front_tokens)
            response = "".join(back_tokens)

            prompts.append(prompt)
            responses.append(response)
            bar.update(1)
            if len(prompts) == N:
                break

    assert len(prompts) == N
    with open(out_path, "w") as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")
    print(f"✅ wrote {out_path} ({N} prompts)")