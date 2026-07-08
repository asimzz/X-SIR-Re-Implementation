"""SemStamp: sentence-level semantic watermark via LSH + rejection sampling.

Port of the LSH variant of Hou et al., "SemStamp: A Semantic Watermark with Paraphrastic
Robustness for Text Generation" (NAACL 2024, https://github.com/abehou/SemStamp), adapted to
this repo's cross-lingual pipeline:

  * Multilingual sentence encoder (paraphrase-multilingual-mpnet-base-v2) instead of the
    English-only SemStamp SBERT, so source languages and back-translation pivots embed
    consistently.
  * LSH hasher reimplemented in NumPy (seed 1234) to drop the fragile `nearpy` dependency.
  * Region-mask RNG made device-independent (the reference hardcodes CUDA) so generation and
    detection agree regardless of device.
  * Generation loop generalized to any causal LM (the reference restricts to OPT).

Detection returns a KGW-style sentence-level z-score AND green/total sentence counts, so it
plugs into detect.py and the existing STEAM gamma_lang scoring path unchanged.
"""
