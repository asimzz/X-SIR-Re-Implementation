from transformers import AutoTokenizer
from collections import Counter
import json
import os
from pathlib import Path

# === CONFIG ===
model_abbr = "llama-3.2-1B"
base_model = "meta-llama/Llama-3.2-1B"
seeds = list(range(50))  # 0 to 49
base_dir = Path("data/mapping/xsir/" + model_abbr)

for ref_seed in seeds:
    print(f"\n== Using seed {ref_seed} as reference ==")

    # Load mapping for reference seed
    mapping_ref = json.load(open(base_dir / f"300_mapping_{model_abbr}_seed{ref_seed}.json"))

    # Load translation output corresponding to reference seed
    translation_file = Path(f"gen/{model_abbr}/xsir/seed_{ref_seed}/mc4.en-bn.mod.z_score.jsonl")
    with open(translation_file) as f:
        all_biases = [json.loads(line)["biases"] for line in f]

    token_strs = [bias[0] for biases in all_biases for bias in biases]
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    token_counts = Counter(token_strs)
    top_10_tokens = [token for token, _ in token_counts.most_common(10)]
    top_10_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in top_10_tokens]

    # Patch all other seeds using reference top-10
    for target_seed in seeds:
        if target_seed == ref_seed:
            continue  # Skip self

        mapping_target = json.load(open(base_dir / f"300_mapping_{model_abbr}_seed{target_seed}.json"))

        for token_id in top_10_token_ids:
            mapping_target[token_id] = mapping_ref[token_id]

        out_path = base_dir / f"ref_seed{ref_seed}" /f"300_mapping_{model_abbr}_seed{target_seed}.json"
        if os.path.dirname(out_path) != "":
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(mapping_target, f, indent=2)
        print(f"  → Patched {target_seed} using {ref_seed}: saved to {out_path}")
