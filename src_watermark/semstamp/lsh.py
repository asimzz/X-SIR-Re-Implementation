"""LSH primitives for SemStamp.

`LSHHasher` partitions the sentence-embedding space with `lsh_dim` random hyperplanes
(seeded so generation and detection agree): a sentence maps to an `lsh_dim`-bit signature ->
an integer region id in [0, 2**lsh_dim). `get_mask_from_seed` deterministically derives the
"green" region set for the next sentence from the previous sentence's region id (this is the
sequential seeding that carries the watermark), exactly as in the reference.
"""
import numpy as np
import torch

# Matches sampling_utils.hash_key in the reference.
hash_key = 15485863

# Device-INDEPENDENT generator for the region-mask permutation. The reference seeds a CUDA
# generator; using a CPU generator here guarantees generation (possibly on GPU) and detection
# produce identical green sets regardless of device.
_mask_rng = torch.Generator()


def get_mask_from_seed(lsh_dim: int, accept_rate: float, seed: int) -> set:
    """Green region ids for a sentence whose predecessor hashed to `seed`.

    Reproduces the reference: permute the 2**lsh_dim region ids with an RNG seeded by
    hash_key * seed, and take the first floor(accept_rate * 2**lsh_dim) as the green list.
    Returned as a Python set for cheap membership tests.
    """
    n_bins = 2 ** lsh_dim
    n_accept = int(n_bins * accept_rate)
    _mask_rng.manual_seed((hash_key * int(seed)) % (2 ** 63 - 1))
    perm = torch.randperm(n_bins, generator=_mask_rng)
    return set(perm[:n_accept].tolist())


class LSHHasher:
    """Random-hyperplane LSH. `normals` is a fixed (lsh_dim, emb_dim) matrix (seed 1234)."""

    def __init__(self, lsh_dim: int, emb_dim: int, seed: int = 1234):
        rs = np.random.RandomState(seed)
        self.normals = rs.randn(lsh_dim, emb_dim).astype(np.float32)
        self.lsh_dim = lsh_dim

    def hash(self, embeds: np.ndarray):
        """embeds: (n, emb_dim). Returns a list of int region ids."""
        embeds = np.asarray(embeds, dtype=np.float32)
        if embeds.ndim == 1:
            embeds = embeds[None, :]
        proj = embeds @ self.normals.T            # (n, lsh_dim)
        bits = (proj > 0).astype(int)             # sign bit per hyperplane
        ids = []
        for row in bits:
            ids.append(int("".join(str(int(b)) for b in row), 2))
        return ids
