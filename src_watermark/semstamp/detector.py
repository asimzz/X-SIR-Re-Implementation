"""SemStamp detection: sentence-level KGW-style z-score.

Segment the text into sentences; for each sentence i>=1, derive the green region set from
sentence i-1's LSH region and check whether sentence i landed in it. With T = n_sentences-1
and lmbd the expected green rate:

    z = (n_green - lmbd*T) / sqrt(T * lmbd * (1-lmbd))

Returns green/total *sentence* counts in the same keys KGW uses (num_green_tokens /
num_tokens_scored) plus green_fraction, so this detector drops into detect.py and the STEAM
gamma_lang scoring path with no special-casing.
"""
import math

from scipy.stats import norm

from src_watermark.semstamp.sentence_model import MultilingualSBERTLSH, DEFAULT_EMBEDDER
from src_watermark.semstamp.sampling_utils import split_sentences
from src_watermark.semstamp.lsh import get_mask_from_seed


class SemStampDetector:
    def __init__(self, embedder_name: str = DEFAULT_EMBEDDER, lsh_dim: int = 3,
                 lmbd: float = 0.25, device: str = None, batch_size: int = 32, seed: int = 1234):
        self.lsh_model = MultilingualSBERTLSH(embedder_name, lsh_dim, device, batch_size, seed)
        self.lsh_dim = lsh_dim
        self.lmbd = lmbd
        # STEAM's gamma_lang fallback reads detector.gamma; the null green-sentence rate is lmbd.
        self.gamma = lmbd

    def detect(self, text):
        sents = split_sentences(text)
        if len(sents) < 2:
            # Mirrors the KGW "too short" signal that detect.py catches -> z_score=None.
            raise ValueError(f"Must have at least 2 sentences to score (got {len(sents)})")

        hashes = self.lsh_model.get_hash(sents)
        T = len(sents) - 1
        n_green = 0
        for i in range(1, len(sents)):
            green = get_mask_from_seed(self.lsh_dim, self.lmbd, hashes[i - 1])
            if hashes[i] in green:
                n_green += 1

        denom = math.sqrt(T * self.lmbd * (1 - self.lmbd))
        z = (n_green - self.lmbd * T) / denom if denom > 0 else float("nan")
        return {
            "z_score": z,
            "p_value": float(norm.sf(z)) if denom > 0 else None,
            "num_green_tokens": n_green,       # green sentences
            "num_tokens_scored": T,            # scored sentences (total - 1)
            "green_fraction": (n_green / T) if T > 0 else float("nan"),
        }
