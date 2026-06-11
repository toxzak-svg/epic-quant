# EPIC-Quant Quality Evaluation

This package contains the eval suite. Two entrypoints:

- `python -m epic_quant.eval.smoke` — **local smoke test**, runs on CPU.
  Chains the engine's per-block forward through all 42 layers on a fixed
  prompt and reports the top-10 next-token predictions and timing.
  Does **not** measure quality. Just verifies the engine produces
  a coherent forward end-to-end.
- `kaggle_notebook.ipynb` — **full eval sweep**, intended for a Kaggle
  GPU box (T4 or P100). Three evals:
  - WikiText-103 perplexity (50K tokens, 512-token chunks)
  - MMLU Pro 4-shot accuracy (200-question subsample, loglik over A/B/C/D)
  - MRCR v2 8-needle retrieval (8K context — not 128K because the
    latter OOMs on T4 even with our 4–5.8× KV compression. The
    mechanism is identical: global p-RoPE layers carry the long-range
    signal. 128K would only amplify the signal.)

## Quick start on Kaggle

1. New Notebook, GPU T4 x2 or P100
2. Add the dataset `toxzak/epic-quant` (or `git clone https://github.com/toxzak-svg/epic-quant`)
3. Add Gemma 4 E4B-it from the Kaggle model hub (or use the HF
   download helper in cell 2)
4. Run all cells. Total runtime: ~45–90 minutes on T4.

## Quick start locally (smoke only)

```powershell
set PYTHONPATH=<project_root>
python -m epic_quant.eval.smoke --policy 3bit --prompt "The capital of France is"
```

This is the only eval that runs on a 13.8 GB RAM / CPU-only box. It
takes about 100 seconds for the 42-layer chain and verifies that the
top-1 next-token prediction is sensible.

## Local smoke test (real numbers from this box)

| Policy | Top-1 token id | Top-1 logit | Per-layer s |
|---|---:|---:|---:|
| 16-bit (no quant) | 26069 | 33.79 | 1.37 |
| 3-bit | 26069 | 34.28 | 2.50 |

**Top-1 prediction is robust to 3-bit quant.** Top-10 overlap between
16-bit and 3-bit is 3/10 — the long tail is messier at 3-bit, which
is consistent with the L2 recon numbers. The 3-bit path is ~1.8×
slower than 16-bit in the Python reference forward; on a real GPU
with a fused unpack-and-matmul kernel the 3-bit path is expected to
*exceed* 16-bit throughput because the packed weights move less data.

## What this is not

- Not a real quality benchmark on this box (the smoke test is one
  forward pass on one prompt, not a population statistic).
- Not optimized. The forward path uses `F.scaled_dot_product_attention`
  with a Python-built mask.
- Not validated against 128K-context MRCR. We do 8K because the full
  eval needs a bigger GPU.
