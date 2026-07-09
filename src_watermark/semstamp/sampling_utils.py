"""Sentence-level generation helpers (ported/adapted from SemStamp sampling_utils.py)."""
import numpy as np
import torch
from transformers import StoppingCriteria
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

MAX_TRIALS = 100
PUNCTS = '!.?'


def split_sentences(text):
    """Shared sentence segmentation for generation and detection. Uses NLTK punkt (as in the
    reference). Punkt is English-trained but handles Latin-script terminators (. ! ?) fine;
    CJK segmentation is weaker (documented caveat)."""
    if isinstance(text, list):
        text = " ".join(text)
    return sent_tokenize(text)


class SentenceEndCriteria(StoppingCriteria):
    """Stop once the decoded text contains one more sentence than when `update` was last
    called. BATCH SIZE 1 ONLY. Uses the same tokenizer as generation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.current_num_sentences = 0

    def update(self, current_text):
        self.current_num_sentences = len(split_sentences(current_text))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.size(0) == 1
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return len(split_sentences(text)) > self.current_num_sentences + 1


def discard_final_token_in_outputs(outputs):
    return outputs[:, :-1]


def gen_sent(model, tokenizer, text_ids, gen_config, stopping_criteria):
    """Generate exactly one more sentence (stopping_criteria fires at the sentence boundary).
    Generalized from the reference (which restricted to OPT models)."""
    outputs = model.generate(
        text_ids,
        generation_config=gen_config,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
    )
    outputs = discard_final_token_in_outputs(outputs)
    new_text_ids = outputs
    new_text = tokenizer.decode(new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
    return new_text, new_text_ids


def reject_close_generation(lsh_model, sents, margin, cutoff=None):
    """Reject sentences whose embedding sits within `margin` cosine of any LSH hyperplane
    (ambiguous region -> fragile under paraphrase/translation). margin=0 accepts everything."""
    if margin <= 0:
        return list(sents), list(range(len(sents)))
    embeds = lsh_model.get_embeddings(sents).astype(np.float32)
    normals = lsh_model.hasher.normals
    if cutoff is not None:
        normals = normals[:cutoff]
    e = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    sims = np.abs(e @ n.T)               # (n_sent, n_normals)
    min_sims = sims.min(axis=1)
    select = [i for i in range(len(min_sims)) if min_sims[i] >= margin]
    return [sents[i] for i in select], select
