"""EPIC-Quant quality evaluation package.

Two entrypoints:
  - `smoke` — runs on CPU, validates the engine produces a sensible
    forward end-to-end. Quick sanity check.
  - `kaggle_notebook` — full eval sweep (WikiText PPL + MMLU Pro +
    MRCR v2 8-needle 128K) intended to run on a Kaggle GPU box.
"""
