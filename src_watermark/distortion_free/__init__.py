"""Distortion-free watermarks (ITS + EXP).

Faithful port of the torch code path of Kuditipudi, Thickstun, Hashimoto & Liang,
"Robust Distortion-free Watermarks for Language Models"
(https://github.com/jthickstun/watermark).

ITS  = inverse-transform sampling watermark  (transform/)
EXP  = exponential-minimum / Gumbel sampling watermark  (gumbel/)

The repo-facing interface lives in `watermark.py`:
  - DistortionFreeGenerator: custom key-driven decode loop (sampling-rule watermark).
  - DistortionFreeDetector:   .detect(text) -> {"z_score": -log(p_value), "p_value": p}.
"""
