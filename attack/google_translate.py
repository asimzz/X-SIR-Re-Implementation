import os
import json
import argparse
from tqdm import tqdm
from deep_translator import GoogleTranslator


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def main(args):
    # Initialize the translator
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    translation_part = args.translation_part

    # Load data
    input_data = read_jsonl(args.input_file)

    output_data = []
    if os.path.exists(args.output_file):
        output_data = read_jsonl(args.output_file)

    print(
        f"Translating {args.translation_part} from {args.src_lang} to {args.tgt_lang}"
    )

    print(f"{len(input_data)} prompts found. {len(output_data)} translations found.")
    if len(input_data) == len(output_data):
        print("Translation already done. Skipping...")
        return

    # Check if the source or target languages are Chinese
    if src_lang == "zh":
        src_lang = "zh-CN"
    if tgt_lang == "zh":
        tgt_lang = "zh-CN"

    # Initialize the translator
    translator = GoogleTranslator(source=src_lang, target=tgt_lang)
    translated_samples = []

    with open(args.input_file, "r", encoding="utf-8") as lines:
        for line in tqdm(
            lines, desc=f"Translating {translation_part}", unit=translation_part
        ):
            data = json.loads(line)
            if translation_part in data:
                translation = translator.translate(data[translation_part])
                data[translation_part] = translation
            translated_samples.append(data)

    # Save the translated prompts to a new JSONL file
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for sample in translated_samples:
            output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Translated file saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument("--src_lang", type=str, required=True, help="Source language")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument(
        "--translation_part",
        type=str,
        required=True,
        help="Part of the JSON to translate",
    )

    args = parser.parse_args()
    main(args)
