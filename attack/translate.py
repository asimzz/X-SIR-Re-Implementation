import os
import json
import asyncio
import requests
import argparse
from langcodes import Language
from call_openrouter import (
    CallOpenRouter,
)  # Updated to use OpenRouter-specific API call handling
from dotenv import load_dotenv

load_dotenv()


def valid_location():
    res = requests.get("https://ipinfo.io", timeout=5).text
    res = json.loads(res)
    country = res.get("country", "")
    print(json.dumps(res, indent=2))
    return country not in ["HK", "CN", "RU"]


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def main(args):
    assert os.getenv(
        "OPENROUTER_API_KEY"
    ), "Set the OPENROUTER_API_KEY environment variable"
    # Load data
    input_data = read_jsonl(args.input_file)

    output_data = []
    if os.path.exists(args.output_file):
        output_data = read_jsonl(args.output_file)

    print(
        f"Translating {args.translate_part} from {args.src_lang} to {args.tgt_lang} using model {args.model}"
    )
    print(f"{len(input_data)} prompts found. {len(output_data)} translations found.")
    if len(input_data[:99]) == len(output_data) or len(input_data[:98]) == len(output_data):
        print("Translation already done. Skipping...")
        return

    def input_to_requests_func(input_file: str, output_file: str) -> list:
        rqs = []
        done_ids = []

        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                if i in done_ids:
                    continue
                prompt = json.loads(line.strip())["prompt"]
                response = json.loads(line.strip())["response"]
                translate_part = json.loads(line.strip())[f"{args.translate_part}"]
                rq = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Translate the following {Language.make(language=args.src_lang).display_name()} text to {Language.make(language=args.tgt_lang).display_name()}:\n\n{translate_part}",
                        }
                    ],
                    "temperature": args.temperature,
                    "metadata": {
                        "row_id": i,
                        "prompt": prompt,
                        "response": response,
                        "translate_part": args.translate_part,
                    },
                }
                rqs.append(rq)
        return rqs[:100]

    def response_to_output_func(response: dict, output_file_path: str):
        translation = response["response"]["choices"][0]["message"]["content"]
        metadata = response["metadata"]
        id = metadata["row_id"]
        prompt = metadata["prompt"]
        prompt_response = metadata["response"]
        translation_part = metadata["translate_part"]
        if translation_part == "prompt":
            prompt = translation
        elif translation_part == "response":
            prompt_response = translation

        json_string = json.dumps(
            {"id": id, "prompt": prompt, "response": prompt_response},
            ensure_ascii=False,
        )
        with open(output_file_path, "a") as f:
            f.write(json_string + "\n")

    def post_run_func(output_file_path: str):
        results = []
        with open(output_file_path, "r") as f:
            for line in f:
                results.append(json.loads(line.strip()))
        results = sorted(results, key=lambda x: x["id"])

        with open(output_file_path, "w") as f:
            for r in results:
                json_string = json.dumps(
                    {"prompt": r["prompt"], "response": r["response"]},
                    ensure_ascii=False,
                )
                f.write(f"{json_string}\n")

    def is_all_done(input_file_path: str, output_file_path: str) -> bool:
        if not os.path.isfile(output_file_path):
            return False

        with open(input_file_path, "r") as f:
            num_requests = len(f.readlines())

        with open(output_file_path, "r") as f:
            num_done = len(f.readlines())

        return num_requests == num_done

    openrouter_caller = CallOpenRouter(
        request_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        input_file_path=args.input_file,
        output_file_path=args.output_file,
        max_attempts=5,
        input_to_requests_func=input_to_requests_func,
        response_to_output_func=response_to_output_func,
        post_run_func=post_run_func,
        is_all_done_func=is_all_done,
    )

    asyncio.run(openrouter_caller.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--src_lang", type=str, required=True, help="Source language")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument(
        "--translate_part", type=str, required=True, help="Part to translate"
    )
    args = parser.parse_args()
    main(args)
