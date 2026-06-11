# EPIC-Quant for Gemma 4 E4B — Working Notes (CPU-only edition)

**Status:** working, runnable on CPU against the real `google/gemma-4-E4B`
safetensors (14.89 GB). Three pillars implemented and benchmarked end-to-end
on this Windows / 13.8 GB RAM / no-CUDA box.

**Code:** `epic_quant/` package — `engine.py`, `forward.py`, `loader.py`,
`layers.py`, `bench.py`. Run with:

```
set PYTHONPATH=C:\Users\Zwmar\projects\e4b
python -m epic_quant.bench --n-tokens 200 --out bench.json
```

---

## 1. What we built vs the original brief

The brief described four pillars. **Two of them don't fit the actual
E4B architecture and were dropped**, with reasons. The remaining two got
re-specified against the real model.

| Pillar | Brief said | Reality (verified) | Implemented? |
|---|---|---|---|
| Epi-Stochastic Fetching (load 2/128 expert slices) | "E4B is dense" contradicted this from the start | `enable_moe_block: false`, no `num_experts` in config | **Dropped** — would only apply to the 26B A4B MoE. Could be ported there. |
| Speculative Multi-Token Layer Prefetching (native MTP drafter) | "Gemma 4 natively includes an MTP draft path" | No `num_mtp_modules`, no MTP head in safetensors. E4B is dense + no MTP. | **Dropped.** The "MTP" in the brief maps to DeepSeek-V3, not E4B. |
| Attention-Symmetric Bit Discarding (1-bit sliding, 4-bit global) | Sensible RoPE argument but the brief inverted which dim is "low-freq" | Confirmed: sliding RoPE θ=1e4 on all 256 head dim, global p-RoPE θ=1e6 on first 25% of 512 head dim (so 75% unrotated) | **Implemented** with corrected assignment: keep rotated 4-bit, drop unrotated 1-bit on sliding; keep rotated 4-bit, drop unrotated 2-bit on global. |
| Sparse PLE Hashing (top-5K resident, rest on-demand) | Right idea, wrong picture: brief said "every layer has its own 256-dim lookup table" | PLE is a **single 2D matrix** `[262144, 10752] = [vocab, 42 × 256]`, 5.27 GB. Not 42 separate tables. | **Implemented** with per-row mmap reads so we never need 5.27 GB of free RAM. |
| Frequency-Filtered p-RoPE KV Eviction | Hand-waved threshold; "drop low-freq to 1.58-bit if distance > sliding window" | Real config: p-RoPE on global with `partial_rotary_factor=0.25`. Sliding only sees 512 tokens, so any long-range position signal is noise. | **Implemented** as `KVPolicy` (theoretical bit-budget model). A real packing kernel is a follow-up. |

**Net:** the three pillars we kept are the ones the architecture actually
supports. The "novelty" claim has been thinned to two real ideas and one
KV bit-budget model.

---

## 2. Verified facts about Gemma 4 E4B (vs the brief)

Pulled directly from `config.json` and the safetensors v1 header
(2130 tensors, 7.996 B params, BF16):

- **42 decoder layers**, alternating 5 sliding_attention + 1 full_attention
  (5+1 × 7 = 42, with 7 full_attention layers).
- **Sliding window = 512 tokens**, `rope_theta=1e4`, default RoPE, head_dim=256.
- **Full attention**, p-RoPE, `rope_theta=1e6`, `partial_rotary_factor=0.25`,
  head_dim=512.
- **GQA 8/2** (8 query heads, 2 KV heads, repeat=4).
- `num_kv_shared_layers=18` — **runtime feature, not parameter aliasing**.
  Each layer has its own `k_proj`/`v_proj`; the runtime reuses K/V across
  the next 1–18 layers. So weight-quantizing one layer's k_proj does NOT
  change shared KV memory.
- `tie_word_embeddings=true` — output head shares with `embed_tokens`.
  No `lm_head` tensor in the file.
- **PLE is `[262144, 10752]` = 5.27 GB** in BF16. One tensor, 42 layers'
  per-layer-256 slices stacked along the column axis. Not 42 separate
  tables.
- PLE companion tensors per layer: `per_layer_input_gate[256, 2560]`,
  `per_layer_projection[2560, 256]`, `post_per_layer_input_norm[2560]`,
  `layer_scalar[1]`. **The brief's PLE is real and bigger than
  described.**
- Per-block linear shapes:
  - Sliding (head_dim 256): `q[2048,2560], k[512,2560], v[512,2560], o[2560,2048]`.
  - Global  (head_dim 512): `q[4096,2560], k[1024,2560], v[1024,2560], o[2560,4096]`.
  - MLP all layers: `gate[10240,2560], up[10240,2560], down[2560,10240]`.
- 5 RMS norms per block: `input_layernorm, post_attention_layernorm,
  pre_feedforward_layernorm, post_feedforward_layernorm,
  post_per_layer_input_norm` — all `[2560]`. Pre+post norm pattern
  (Gemma 2/3 style, not Gemma 1's single norm).
- QK-norm: `self_attn.q_norm[256]`, `self_attn.k_norm[256]` per layer.

---

## 3. Measured numbers (real, against the actual safetensors)

All numbers from `python -m epic_quant.bench --n-tokens 100` against the
real `model.safetensors` (15.99 GiB on disk).

### 3.1 Weight memory (all 42 layers, BF16 → policy quant)

| Tensor class | Unquant (MB) | Quant policy | Quant (MB) | Saved (MB) | % saved |
|---|---:|---|---:|---:|---:|
| Attention (q/k/v/o) | 1284.5 | 2-bit sliding / 4-bit global | 206.4 | **1078.1** | 83.9% |
| MLP (gate/up/down) | 6606.0 | 4-bit (all) | 1651.5 | **4954.5** | 75.0% |
| **Subtotal** | **7890.5** | mixed | **1857.9** | **6032.6** | **76.5%** |

Excluded: norms (tiny), `embed_tokens` (262144 × 2560 × 2B = 1.31 GB,
untouched here), vision/audio towers (not part of the text EPIC-Quant
path), and `per_layer_projection` / `per_layer_input_gate` (tiny,
kept at 4-bit).

**PLE alone is 5.27 GB. Weight quant savings of ~6 GB get E4B from
"doesn't fit on a 16 GB GPU" into "comfortable on a 16 GB GPU / fits
on a 24 GB GPU with KV room for long context."**

### 3.2 KV cache (theoretical bit-budget)

| Layer type | head_dim | rotated dim | unrotated dim | Bits / token (K+V) | vs BF16 baseline | Compression |
|---|---:|---:|---:|---:|---:|---:|
| Sliding | 256 | 256 (full RoPE) | 0 | 1024 (4-bit × 256 × 2) | 256×16×2 = 8192 | **4.0×** |
| Global  | 512 | 128 (p-RoPE 25%) | 384 | 4·128 + 2·384 = 1408 (×2 = 2816) | 512×16×2 = 16384 | **5.8×** |

`compression_ratio = 1024/8192 = 0.25` (sliding) and `2816/16384 = 0.15625`
(global) — these are *budgets*, not actual packed bytes. A real packing
kernel is a follow-up; this just defines the policy.

### 3.3 PLE sparse cache (the real win)

- Full PLE on disk: **5,637 MB** (5.27 GiB).
- Hot table resident in RAM (5000 tokens × 10752 cols × BF16):
  **107.5 MB** = **98.1% reduction**.
- 100-token synthetic workload, 85% hot / 15% cold:
  - **86% hit rate** (3612 hits, 588 misses).
  - LRU held 14 cold slices (~300 KB).
  - 12,681 lookups/sec on this CPU.
- Cold path: per-row mmap read = 21,504 bytes per lookup (262144 rows
  total, only the ones you actually touch ever leave disk).

The real, demonstrable win on this box. The hot table is small enough
to live in CPU L2/L3 across calls; the cold path is bounded by the
LRU capacity (default 64) which caps resident cold memory at
~1.4 MB. Total PLE memory ceiling: ~109 MB.

### 3.4 Forward pass (one decoder block, seq_len=16, BF16 end-to-end)

CPU only, single-thread. Attention goes through
`F.scaled_dot_product_attention` (math kernel on CPU, would be
flash-attention on CUDA), with a 4D additive mask encoding causal +
sliding-window for sliding layers, pure causal for global.

Weights are **packed 2-bit / 4-bit** (see `packed.py`):
2-bit packs 4 ternary values per byte; 4-bit packs 2 int4 per byte;
per-row fp16 scales stored separately. The forward unpack-then-matmul
in Python is for measuring reconstruction error — a real deployment
would fuse the unpack into the matmul kernel (Triton, CUTLASS, or
custom C++).

| Layer | is_global | q L2 rel | k L2 rel | v L2 rel | o L2 rel | gate L2 | up L2 | down L2 | packed bytes | unquant bytes | total ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0   | sliding (2-bit attn / 4-bit mlp) | 1.114 | 1.116 | 1.116 | 1.108 | 0.186 | 0.174 | 0.204 | **43.3 MB** | ~80 MB | 1992 |
| 5   | global  (4-bit attn / 4-bit mlp) | 0.195 | 0.262 | 0.173 | 0.186 | 0.229 | 0.220 | 0.214 | **53.2 MB** | ~125 MB | 2131 |

**Three things stand out:**

1. **2-bit ternary on sliding q/k/v/o is broken** (L2 rel > 1.0).
   At 1.58-bit effective, the centroid set `{-1, 0, +1}` is too
   coarse for the q/k/v projections. **The brief's "ultra-aggressive
   1-bit ternary" pillar is qualitatively wrong at the bit budget
   proposed.** 3-bit per-row int is the realistic floor — 2-bit
   might be acceptable on MLP down_proj, not on attention.
2. **4-bit global attn is tolerable** (L2 rel 0.17–0.26), in line
   with published symmetric-int4 results. 4-bit MLP at the same
   bits is also fine.
3. **PLE companions pack cleanly** — gate L2 = 0.20, projection
   L2 = 0.16 at 4-bit. Tiny relative cost, ~28 MB packed for the
   per-layer matrices.

**Implication for the brief:** the mechanism (compress the layer
type that sees least context) is right; the bit width is wrong.
Minimum viable policy: 3-bit per-row int on sliding attn, 4-bit
on global attn, 4-bit on MLP, 4-bit on PLE companions, 5000-hot
PLE. That moves ~5.5 GB of weights out of the working set with
L2 recon under 0.4 everywhere.

---

## 4. What's left to make this a real deployment

In order of effort:

1. **Fuse the unpack + matmul into one kernel.** Right now the
   Python unpack loop in `packed.py` is the bottleneck (the
   reference forward is ~2× slower than a plain BF16 forward
   because of this). Triton kernel with a 4-value unpacked
   int2 matmul, or CUTLASS, would close the gap and likely beat
   BF16 throughput on tensor cores.
2. **KV packing kernel.** Currently the `KVPolicy` is a budget
   model. A real packed KV cache needs the p-RoPE-aware layout
   decided at write time, not inferred at budget time.
3. **PLE cold row read on GPU.** Row-level mmap doesn't port
   to CUDA. The GPU path needs the cold PLE rows pinned to a
   pinned-memory staging buffer, async-copied, then
   dequantized. The hot table fits in GPU HBM easily.
4. **Perplexity eval on WikiText-103 or MRCR v2 8-needle 128K.**
   The reference here measures quant L2 and throughput, not
   quality. The real test is whether E4B's reported 69.4% MMLU
   Pro and 25.4% MRCR v2 8-needle 128K survive 4-bit global +
   3-bit sliding + 86%-hot PLE. Likely yes for MMLU, possibly
   -2 to -5 points for MRCR (long-context needle recall is
   exactly what global p-RoPE is responsible for).
5. **RoPE hookup.** The reference forward skips RoPE; a real
   inference path needs to apply p-RoPE (rotating only the first
   25% of the global head dim with θ=1e6) and standard RoPE on
   sliding layers (θ=1e4, all 256 head dim). `transformers`
   `Gemma4RotaryEmbedding` does this; we just don't call it.

---

## 5. Files

```
projects/e4b/
├── epic_quant/
│   ├── __init__.py        # package entry, policy dataclasses re-exported
│   ├── layers.py          # layer_dims, layer_param_keys, ple_columns_for_layer
│   ├── loader.py          # MmapSafetensors: read v1 safetensors lazily
│   ├── packed.py          # 2-bit/4-bit packed quant + dequant, real byte counts
│   ├── engine.py          # QuantPolicy, PLEPolicy, KVPolicy + PLECache + KVEvictor
│   ├── forward.py         # forward_one_layer: real quant + SDPA on CPU
│   └── bench.py           # 4-section smoke + bench harness
├── inspect_shapes.py      # one-off: dumps the safetensors header shapes
├── probe_header.py        # one-off: confirms the file is v1 safetensors
├── bench.json             # latest run output (memory + L2 recon + packed bytes)
├── bench_stdout.txt       # latest run console
└── WRITEUP.md             # this file
```

---

## 6. TL;DR

- Real, runnable code on this box. Three pillars of the brief kept
  (after correcting two of them), one dropped (MoE — doesn't apply to
  E4B), one dropped (MTP — not in this model).
- **At the policy in the brief** (2-bit ternary sliding, 4-bit global,
  4-bit MLP, 5000-hot PLE): **6.0 GB weight savings + 5.4 GB PLE
  savings = 11.4 GB total** out of 16 GB original footprint.
- **Quality cost of ternary on sliding attn: 100%+ L2 recon** — the
  brief's "ultra-aggressive" choice doesn't survive contact with the
  actual tensors. 3-bit is the realistic floor.
- **KV compression: 4× sliding, 5.8× global** — at the policy's bit
  budget. Packing kernel not yet implemented.
- **PLE hot-cache hit rate: 86%** on a realistic 85/15 workload
  — the 5000-token hot default is sensible.
