"""Repo-facing wrappers for the distortion-free (ITS/EXP) watermarks.

These adapt the vendored jthickstun/watermark code to the loose "unified interface" the rest
of this codebase expects:

  * DistortionFreeGenerator.generate(model, prompt_ids, m) -> token ids
        used by gen.py in place of a HuggingFace LogitsProcessor (ITS/EXP are sampling-rule
        watermarks and cannot be expressed as a logit bias).

  * DistortionFreeDetector.detect(text) -> {"z_score": -log(p_value), "p_value": p}
        matches the detector contract in detect.py (a dict carrying a higher-=-watermarked
        "z_score"), so eval_detection.py / evaluate_by_tier.py and the STEAM output schema
        work unchanged. We put -log(p_value) in the "z_score" slot by design.

Two detection modes:
  * exact  (null_results is None): faithful per-text permutation_test (slow). k defaults to
        the full token length (as in the reference demo/detect.py) — self-consistent per text.
  * fast   (null_results given):   fast_permutation_test against a precomputed, sorted null
        distribution of the test statistic (built by precompute_null.py from human validation
        text in the relevant language). Requires a FIXED k so statistics are comparable across
        texts; STEAM uses this mode.
"""
import math

import numpy as np
import torch

from src_watermark.distortion_free.generation import generate
from src_watermark.distortion_free.detection import (
    phi,
    permutation_test,
    fast_permutation_test,
)
from src_watermark.distortion_free.transform.key import transform_key_func
from src_watermark.distortion_free.transform.sampler import transform_sampling
from src_watermark.distortion_free.gumbel.key import gumbel_key_func
from src_watermark.distortion_free.gumbel.sampler import gumbel_sampling

VALID_METHODS = ("its", "exp")


def _resolve_vocab_size(tokenizer, model=None):
    """Vocab size must be identical at generation and detection. Prefer the tokenizer length
    (what detection sees); fall back to the model config for generation if needed."""
    if tokenizer is not None:
        return len(tokenizer)
    if model is not None:
        return model.config.vocab_size
    raise ValueError("Cannot resolve vocab_size without a tokenizer or model")


class DistortionFreeGenerator:
    """Custom key-driven decode loop for ITS/EXP (distortion-free) generation."""

    def __init__(self, method, key, n, vocab_size):
        method = method.lower()
        if method not in VALID_METHODS:
            raise ValueError(f"Unknown distortion-free method: {method}")
        self.method = method
        self.key = int(key)
        self.n = int(n)
        self.vocab_size = int(vocab_size)
        if method == "its":
            self.key_func, self.sampler = transform_key_func, transform_sampling
        else:  # exp
            self.key_func, self.sampler = gumbel_key_func, gumbel_sampling

    def generate(self, model, prompt_ids, m):
        """prompt_ids: LongTensor [batch, L]. Returns full sequences (prompt + m new tokens),
        cpu LongTensor [batch, L+m]. A single shared key is used for every text in the batch
        (like KGW's fixed seed) so detection can reproduce the key sequence."""
        batch_size = prompt_ids.shape[0]
        seeds = [self.key] * batch_size
        return generate(
            model,
            prompt_ids,
            self.vocab_size,
            self.n,
            m,
            seeds=seeds,
            key_func=self.key_func,
            sampler=self.sampler,
            random_offset=True,  # detection-invariant: adjacency() tries every offset j
        )


class DistortionFreeDetector:
    """Permutation-test detector exposing a higher-=-watermarked `z_score` (= -log p)."""

    def __init__(self, method, key, n, k, gamma, tokenizer, null_results=None, n_runs=100):
        method = method.lower()
        if method not in VALID_METHODS:
            raise ValueError(f"Unknown distortion-free method: {method}")
        self.method = method
        self.key = int(key)
        self.n = int(n)
        self.k = k  # None => use len(tokens) (exact path only)
        self.gamma = float(gamma)
        self.tokenizer = tokenizer
        self.vocab_size = _resolve_vocab_size(tokenizer)
        self.n_runs = int(n_runs)
        self.set_null_results(null_results)

    def set_null_results(self, null_results):
        """Set (or clear) the fast-path null distribution. Accepts a 1-D array/tensor of null
        test-statistic values; stored sorted ascending as a float32 tensor."""
        if null_results is None:
            self.null_results = None
            return
        t = torch.as_tensor(np.asarray(null_results, dtype=np.float32))
        self.null_results, _ = torch.sort(t.flatten())

    # -- test statistic (faithful to experiments/c4-experiment.py bindings) --
    def _make_test_stat(self):
        gamma = self.gamma
        if self.method == "its":
            from src_watermark.distortion_free.transform.score import transform_edit_score

            def dist(x, y):
                return transform_edit_score(x, y, gamma=gamma)

            def test_stat(tokens, n, k, generator, vocab_size, null=False):
                return phi(tokens=tokens, n=n, k=k, generator=generator,
                           key_func=transform_key_func, vocab_size=vocab_size,
                           dist=dist, null=False, normalize=True)
        else:  # exp
            from src_watermark.distortion_free.gumbel.score import gumbel_edit_score

            def dist(x, y):
                return gumbel_edit_score(x, y, gamma=gamma)

            def test_stat(tokens, n, k, generator, vocab_size, null=False):
                return phi(tokens=tokens, n=n, k=k, generator=generator,
                           key_func=gumbel_key_func, vocab_size=vocab_size,
                           dist=dist, null=null, normalize=False)
        return test_stat

    def _encode(self, text):
        tokens = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048)[0]
        return tokens

    def _resolve_k(self, n_tokens):
        if self.k is not None:
            return int(self.k)
        return int(n_tokens)  # exact path default: whole-sequence single block

    def compute_statistic(self, text):
        """Return the raw test statistic for `text` under the fixed key (no permutation).
        Used by precompute_null.py to build the per-language null distribution."""
        tokens = self._encode(text)
        k = self._resolve_k(len(tokens))
        if len(tokens) < k or len(tokens) < 1:
            raise ValueError(f"Must have at least {k} tokens to score (got {len(tokens)})")
        test_stat = self._make_test_stat()
        generator = torch.Generator()
        generator.manual_seed(self.key)
        result = test_stat(tokens=tokens, n=self.n, k=k, generator=generator,
                           vocab_size=self.vocab_size)
        return float(result)

    def detect(self, text):
        tokens = self._encode(text)
        k = self._resolve_k(len(tokens))
        if len(tokens) < k or len(tokens) < 1:
            # Mirrors the KGW/X-SIR "too short" signal that detect.py catches -> z_score=None.
            raise ValueError(f"Must have at least {k} tokens to score (got {len(tokens)})")

        test_stat = self._make_test_stat()
        if self.null_results is not None:
            if self.k is None:
                raise ValueError("fast path requires a fixed --wm_k so statistics are comparable")
            p = fast_permutation_test(tokens, self.vocab_size, self.n, k, self.key,
                                      test_stat, self.null_results)
            floor = 1.0 / (len(self.null_results) + 1)
        else:
            p = permutation_test(tokens, self.vocab_size, self.n, k, self.key,
                                 test_stat, n_runs=self.n_runs)
            floor = 1.0 / (self.n_runs + 1)

        p = max(float(p), floor)  # clamp so -log(p) stays finite when p == 0
        return {"z_score": -math.log(p), "p_value": p}
