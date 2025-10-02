import os
import json
import requests
import argparse
from tqdm import tqdm
from langcodes import Language
from concurrent.futures import ThreadPoolExecutor


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def translate_single(text, src_lang, tgt_lang):
    """Simple sync translation - fast and reliable"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ Please set the GEMINI_API_KEY environment variable.")

    prompt = f"Translate the following {Language.make(language=src_lang).display_name()} text to {Language.make(language=tgt_lang).display_name()}. Just give the final translated text, don't add any prefix messages:\n\n{text}"

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {"temperature": 0.0}
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return f"[ERROR_{response.status_code}]"
    except Exception as e:
        return f"[ERROR] {str(e)[:50]}"


def main(args):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ Please set the GEMINI_API_KEY environment variable.")

    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    part = args.translation_part

    input_data = read_jsonl(args.input_file)

    # Check already done
    done_count = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            done_count = len(f.readlines())

    total = len(input_data)
    print(f"Translating '{part}' from {src_lang} to {tgt_lang}")
    print(f"{total} samples found. {done_count} already translated.")

    if done_count >= total:
        print("✅ Translation already complete.")
        return

    # Use ThreadPoolExecutor for true parallelism
    max_workers = 16  # Increase for better throughput

    def process_item(i):
        data = input_data[i].copy()
        if part in data:
            translated = translate_single(data[part], src_lang, tgt_lang)
            data[part] = translated
        return data

    # Process remaining items with threads
    remaining_indices = list(range(done_count, total))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(remaining_indices), desc="Translating", unit="item") as pbar:
            # Submit all tasks
            future_to_index = {executor.submit(process_item, i): i for i in remaining_indices}

            # Collect results as they complete
            results = {}
            for future in future_to_index:
                try:
                    index = future_to_index[future]
                    result = future.result()
                    results[index] = result
                    pbar.update(1)
                except Exception as e:
                    print(f"⚠️ Error processing index {future_to_index[future]}: {e}")
                    pbar.update(1)

    # Write results in order
    with open(args.output_file, "a", encoding="utf-8") as f:
        for i in sorted(results.keys()):
            f.write(json.dumps(results[i], ensure_ascii=False) + "\n")

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
