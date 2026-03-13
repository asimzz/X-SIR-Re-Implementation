import os
import json
import argparse
from tqdm import tqdm
from langcodes import Language
from g4f.client   import Client
from g4f.Provider import You

client = Client(
    provider = You
)


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def g4f_translate(text, src_lang, tgt_lang, model="gpt-4o", temperature=1.3):
    # Format prompt
    prompt = (
        f"Translate the following {Language.make(language=src_lang).display_name()} text "
        f"to {Language.make(language=tgt_lang).display_name()}:\n\n{text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    """
    You are an expert multilingual translator. Translate the user's input text from the source language to the target language accurately,
    preserving meaning, tone, and context. Do not explain, comment, or add anything extra—only return the translated text.
                    """.strip()
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


def main(args):
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    part = args.translation_part

    input_data = read_jsonl(args.input_file)
    translated_data = read_jsonl(args.output_file) if os.path.exists(args.output_file) else []

    total = len(input_data)
    done = len(translated_data)
    print(f"Translating '{part}' from {src_lang} to {tgt_lang}")
    print(f"{total} samples found. {done} already translated.")

    if done >= total:
        print("✅ Translation already complete.")
        return

    with open(args.output_file, "a", encoding="utf-8") as f_out:
        for idx in tqdm(range(done, total), desc="Translating", unit="line"):
            data = input_data[idx]
            if part in data:
                try:
                    translated = g4f_translate(data[part], src_lang, tgt_lang)
                    data[part] = translated
                except Exception as e:
                    print(f"⚠️ Error on index {idx}: {e}")
                    data[part] = data[part]
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ Translation written to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--src_lang", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--translation_part", type=str, default="response")
    args = parser.parse_args()
    main(args)
