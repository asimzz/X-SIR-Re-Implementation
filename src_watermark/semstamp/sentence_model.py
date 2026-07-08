"""Multilingual sentence encoder + LSH, shared by SemStamp generation and detection."""
import numpy as np
from sentence_transformers import SentenceTransformer

from src_watermark.semstamp.lsh import LSHHasher

DEFAULT_EMBEDDER = "paraphrase-multilingual-mpnet-base-v2"


class MultilingualSBERTLSH:
    """Wraps a multilingual SentenceTransformer with a fixed LSH hasher.

    Generation and detection MUST use the same (embedder, lsh_dim, seed) so the region ids
    line up. The hasher is seeded (default 1234), so a fresh instance reproduces the same
    hyperplanes every time.
    """

    def __init__(self, embedder_name: str = DEFAULT_EMBEDDER, lsh_dim: int = 3,
                 device: str = None, batch_size: int = 32, seed: int = 1234):
        self.embedder = SentenceTransformer(embedder_name, device=device)
        self.emb_dim = self.embedder.get_sentence_embedding_dimension()
        self.hasher = LSHHasher(lsh_dim, self.emb_dim, seed=seed)
        self.lsh_dim = lsh_dim
        self.batch_size = batch_size

    def get_embeddings(self, sents) -> np.ndarray:
        return np.asarray(self.embedder.encode(list(sents), batch_size=self.batch_size))

    def get_hash(self, sents):
        """sents: list[str] -> list[int] region ids."""
        return self.hasher.hash(self.get_embeddings(sents))
