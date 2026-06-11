# EPIC-Quant for Gemma 4 E4B

CPU-first reference implementation of three layers-aware compression
pillars for Google's `gemma-4-E4B` (8 B parameters, 4.5 B effective
with PLE, 42 layers, hybrid sliding-window + global attention with
p-RoPE, dense, no MTP). Measured against the actual safetensors on
disk, no synthetic weights.

**Status: research artifact, not a production inference engine.**
This is a measurement harness with real numbers. It is suitable for
reproducing the measurements, discussion, and as a starting point for
a real deployment (see "What's not here" below).

## What this is

Three pillars, each implemented and benchmarked end-to-end:

1. **Layer-type-aware weight quantization** — sliding-attn
   `q/k/v/o` quantize at one bit budget, global-attn `q/k/v/o` at
   another, MLP and PLE companions at a third. Packed bytes are
   reported as the real on-RAM cost.
2. **PLE (Per-Layer Embedding) sparse hash** — the 5.27 GB
   `[262144, 10752]` PLE table is sparse-cached with a hot top-K
   in RAM and per-row mmap reads for cold tokens. Measured 86% hot
   hit rate on a realistic 85/15 workload.
3. **p-RoPE-aware KV cache eviction budget** — sliding layers keep
   4-bit rotated / drop 1-bit unrotated; global layers keep 4-bit
   rotated / drop 2-bit unrotated (because p-RoPE rotates only 25%
   of the head dim on global). Bit-budget model only — the packing
   kernel is a follow-up.

## What this is not

- Not a `from_pretrained`-able quantized model on HF Hub.
- Not a `transformers` / `vllm` / `llama.cpp` plugin.
- Not validated against MMLU Pro / MRCR v2 8-needle 128K / Codeforces
  ELO. The reference measures **quant L2 reconstruction error and
  forward timing**, not task quality.
- Not optimized. Forward path uses `F.scaled_dot_product_attention`
  with a Python-built mask on CPU. Memory-bandwidth-bound workloads
  on a real GPU with a fused unpack-and-matmul kernel (Triton /
  CUTLASS / custom C++) would beat FP16 throughput at 1.58 and 3 bit.

## The headline finding

The brief's "1.58-bit ternary on sliding attention" pillar is
**qualitatively wrong at the proposed bit budget**. Measured L2
reconstruction error on the actual E4B weights is **>1.0**, which
means the dequantized weights are mostly noise. The mechanism
(compress the low-context layer type) is correct; the bit width
is not.

**3-bit on sliding attn is the realistic floor.** L2 recon drops
from 1.11 → 0.29 (4× improvement) for +114 MB of attn weight
(+6%). 4-bit uniform is the safe conservative choice. Full sweep
in [`COMPARISON.md`](COMPARISON.md), full reasoning in
[`WRITEUP.md`](WRITEUP.md).

## Repo layout

```
epic_quant/
  __init__.py
  layers.py     # layer_dims, layer_param_keys
  loader.py     # MmapSafetensors: lazy v1-safetensors read
  packed.py     # 2-bit / 3-bit / 4-bit / 16-bit packed weight formats
  engine.py     # policies + PLECache + KVEvictor + EPICQuantEngine
  forward.py    # one-block forward (packed quant + real SDPA) on CPU
  bench.py      # single-policy bench and --sweep 4-policy comparison
  build_report.py # turns sweep.json into a markdown table
scripts/
  inspect_shapes.py  # dumps the safetensors header shapes
  probe_header.py    # confirms the file is v1 safetensors
COMPARISON.md   # 1.58 / 3 / 4 / 16-bit sweep, side-by-side
WRITEUP.md      # full architecture writeup, what was built / dropped
LICENSE         # Apache 2.0
```

## How to run

```powershell
# Python 3.10+ with torch, transformers, safetensors, numpy installed.
# CPU is fine; this whole bench runs in 2-5 minutes on a single core.

# 1. Make sure you have a Gemma 4 E4B safetensors somewhere. Either:
#    - download via LM Studio (easiest on this box), or
#    - python -c "from huggingface_hub import snapshot_download;
#                  snapshot_download('google/gemma-4-E4B',
#                                   allow_patterns=['*.json','*.safetensors','tokenizer*'])"

# 2. Run the sweep:
$env:PYTHONPATH = "C:\Users\Zwmar\projects\e4b"
python -m epic_quant.bench --sweep --out sweep.json

# 3. Build the human report:
python -m epic_quant.build_report sweep.json COMPARISON.md
```

Single-policy run (the brief's exact config):

```powershell
python -m epic_quant.bench --sliding-bits 2 --global-bits 4 --mlp-bits 4 `
                           --ple-hot 5000 --out bench.json
```

## Measured numbers (real, this box)

All numbers from `python -m epic_quant.bench --sweep` on the actual
`google/gemma-4-E4B` safetensors (15.99 GiB on disk), CPU, BF16
end-to-end. 200 tokens, seq_len=16, packed 2/3/4-bit weights.

| Policy | Attn | MLP | PLE companions | PLE hot | **Total** | Sliding attn L2 |
|---|---:|---:|---:|---:|---:|---:|
| **1.58-bit (brief)** | 207 MB | 1.65 GB | 28 MB | 108 MB | **1.99 GB** | **1.11** |
| **3-bit** | 322 MB | 1.65 GB | 28 MB | 108 MB | **2.11 GB** | **0.29** |
| **4-bit uniform** | 322 MB | 1.65 GB | 28 MB | 108 MB | **2.11 GB** | 0.17 |
| **16-bit (no quant)** | 1.28 GB | 6.61 GB | 110 MB | 108 MB | **8.11 GB** | 0.00 |

PLE full on disk is 5.27 GB. PLE sparse hash is the second big win
(5.27 GB → 108 MB hot table) and is policy-independent. KV cache
compression (sliding 4×, global 5.8× at the configured bit budget)
is the same across all four policies.

## What's not here (and why)

- **No GPU kernel.** CPU-only. Fused unpack-and-matmul on a real
  GPU is where the throughput win lives.
- **No `transformers` integration.** This is a standalone
  measurement harness, not a model class.
- **No quality eval.** No WikiText-103 PPL, no MMLU Pro, no
  MRCR v2 8-needle 128K. Only quant L2 recon and CPU forward
  time. To make this a real product you would run those evals
  at 1.58 / 3 / 4 bit and confirm L2 recon is a useful proxy
  for the published 69.4% MMLU Pro / 25.4% MRCR.
- **No KV packing kernel.** `KVPolicy` is a bit-budget model
  with theoretical compression ratios. The bytes-on-disk packing
  is a follow-up.
- **No RoPE in the reference forward.** We skip p-RoPE; a real
  deployment would call `transformers`' `Gemma4RotaryEmbedding`.
- **Dropped from the original brief** with reasons documented in
  `WRITEUP.md` §1: Epi-Stochastic Fetching (E4B is dense, not
  MoE), Speculative MTP Prefetching (E4B has no MTP head in
  config or safetensors).

## License

Apache 2.0. See [`LICENSE`](LICENSE). The Gemma 4 E4B weights are
not bundled; they are downloaded at runtime from
`huggingface.co/google/gemma-4-E4B` and remain subject to Google's
Gemma Terms of Use.
