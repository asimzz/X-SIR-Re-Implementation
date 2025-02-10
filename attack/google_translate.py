import json
import argparse
from tqdm import tqdm
from deep_translator import GoogleTranslator


def main(args):
    # Initialize the translator
    translator = GoogleTranslator(source="en", target="zh-CN")

    translated_samples = []

    with open(args.input_file, "r", encoding="utf-8") as lines:
        translation_part = args.translation_part
        for line in tqdm(lines, desc=f"Translating {translation_part}", unit=translation_part):
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
        type=["prompt, response"],
        required=True,
        help="Part of the JSON to translate",
    )

    args = parser.parse_args()
    main(args)
