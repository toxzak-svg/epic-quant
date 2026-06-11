# EPIC-Quant (CPU, Gemma 4 E4B)

Epi-Stochastic Predictive Fetching & Context-Aware Bit-Shifting for
Gemma 4 E4B. Three pillars, CPU-first, real weights.

See [WRITEUP.md](WRITEUP.md) for the full breakdown of what was built,
what was dropped, and the measured numbers.

## Quick start

```powershell
# Download E4B safetensors (or use LM Studio's cache at the path below)
$env:PYTHONPATH = "C:\Users\Zwmar\projects\e4b"
python -m epic_quant.bench --n-tokens 200 --out bench.json
```

## What runs

`bench.py` reports four things against the real `model.safetensors`:

1. Per-layer weight memory, before and after the EPIC-Quant policy.
2. KV cache theoretical compression (sliding 4×, global 5.8×).
3. PLE sparse-cache hot/cold hit rate on a synthetic workload.
4. Forward-pass quant round-trip: L2 recon error and ms/block for a
   sliding layer and a global layer.

## Policies you can tune

- `QuantPolicy(bits_sliding_attn=2, bits_sliding_mlp=4, bits_global_attn=4,
  bits_global_mlp=4, bits_ple_per_layer=4)`
- `PLEPolicy(hot_token_topk=5000, cold_strategy="lazy", lru_capacity=64)`
- `KVPolicy(sliding_unrotated_bits=1, sliding_rotated_bits=4,
  global_unrotated_bits=2, global_rotated_bits=4)`

## Repo layout

```
epic_quant/
  __init__.py
  layers.py     # layer_dims, layer_param_keys
  loader.py     # MmapSafetensors: lazy v1-safetensors read
  packed.py     # 2-bit/4-bit packed quant + dequant, real byte counts
  engine.py     # policies + PLECache + KVEvictor + EPICQuantEngine
  forward.py    # one-block forward (packed quant + SDPA) on CPU
  bench.py      # 4-section bench harness
inspect_shapes.py  # one-off header inspector
probe_header.py    # one-off magic-byte probe
WRITEUP.md
```
