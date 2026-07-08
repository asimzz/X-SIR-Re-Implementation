"""SemStamp watermarked generation via sentence-level rejection sampling.

For each new sentence, the set of "green" LSH regions is seeded by the previous sentence's
region id. We sample a candidate sentence, embed+hash it, and accept iff it lands in a green
region (and passes the margin filter); otherwise we resample, up to MAX_TRIALS, after which we
keep the last candidate (as in the reference). Adapted from SemStamp's lsh_reject_completion:
generalized to any causal LM and made device-independent.
"""
import torch
from transformers import GenerationConfig, StoppingCriteriaList

from src_watermark.semstamp.lsh import get_mask_from_seed
from src_watermark.semstamp.sampling_utils import (
    MAX_TRIALS,
    SentenceEndCriteria,
    gen_sent,
    reject_close_generation,
)


class SemStampGenerator:
    def __init__(self, model, tokenizer, lsh_model, lsh_dim,
                 lmbd=0.25, margin=0.0, max_new_tokens=205,
                 temperature=0.7, top_k=0, repetition_penalty=1.05):
        self.model = model
        self.tokenizer = tokenizer
        self.lsh_model = lsh_model
        self.lsh_dim = lsh_dim
        self.lmbd = lmbd
        self.margin = margin
        self.max_new_tokens = max_new_tokens
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    def generate(self, prompt: str) -> str:
        """Return prompt + watermarked continuation (~max_new_tokens new tokens)."""
        device = self.model.device
        sent_end_criteria = SentenceEndCriteria(self.tokenizer)

        lsh_seed = self.lsh_model.get_hash([prompt])[0]
        accept_mask = get_mask_from_seed(self.lsh_dim, self.lmbd, lsh_seed)

        text = prompt
        text_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_length = text_ids.size(1)
        sent_end_criteria.update(prompt)

        current_trials = 0
        while True:
            stopping = StoppingCriteriaList([sent_end_criteria])
            new_text, new_text_ids = gen_sent(
                self.model, self.tokenizer, text_ids, self.gen_config, stopping)
            if new_text == '':
                break
            current_trials += 1

            # Margin filter (no-op when margin<=0).
            accepted, _ = reject_close_generation(self.lsh_model, [new_text], self.margin)
            if len(accepted) == 0 and current_trials < MAX_TRIALS:
                continue

            candidate = self.lsh_model.get_hash([new_text])[0]
            if candidate in accept_mask or current_trials >= MAX_TRIALS:
                # Accept this sentence; its region seeds the next green set.
                current_trials = 0
                lsh_seed = candidate
                accept_mask = get_mask_from_seed(self.lsh_dim, self.lmbd, lsh_seed)
                text += new_text
                text_ids = new_text_ids
                sent_end_criteria.update(text)
                if (text_ids.size(1) - prompt_length) >= self.max_new_tokens - 1:
                    break
            # else: reject and resample (same context, new sampling randomness)

        return text
