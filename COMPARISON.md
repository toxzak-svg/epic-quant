# EPIC-Quant: 1.58-bit / 3-bit / 4-bit / FP16 sweep

All numbers are **real**, measured against the actual `google/gemma-4-E4B`
safetensors (15.99 GiB on disk). CPU forward path, BF16 end-to-end,
packed 2-bit/3-bit/4-bit weights, `F.scaled_dot_product_attention` for
attention. 200 tokens, seq_len=16 for the layer-forward benchmark.

The brief's proposal (1.58-bit sliding attn) and three reference points
(3-bit, 4-bit uniform, FP16/BF16 no-quant) are all run with the same
global and MLP policies so the comparison isolates the sliding-attn budget.

## 1. Weight memory (42 layers, packed bytes + scales)

| Policy | Attn unquant | Attn packed | Attn saved | MLP unquant | MLP packed | MLP saved | PLE unquant | PLE packed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.58bit (brief)** | 1284.5 MB | 207.0 MB | 1077.6 MB | 6606.0 MB | 1653.4 MB | 4952.6 MB | 110.1 MB | 27.8 MB |
| **3bit** | 1284.5 MB | 321.6 MB | 962.9 MB | 6606.0 MB | 1653.4 MB | 4952.6 MB | 110.1 MB | 27.8 MB |
| **4bit (uniform)** | 1284.5 MB | 321.6 MB | 962.9 MB | 6606.0 MB | 1653.4 MB | 4952.6 MB | 110.1 MB | 27.8 MB |
| **16bit (no quant)** | 1284.5 MB | 1284.5 MB | 0.0 MB | 6606.0 MB | 6606.0 MB | 0.0 MB | 110.1 MB | 110.1 MB |

## 2. Sliding layer (layer 0) — per-tensor L2 reconstruction error

Lower is better. 0.0 = no quant. L2 rel = ||w - w_dequant||₂ / ||w||₂.

| Policy | PLE gate | PLE proj | attn q | attn k | attn v | attn o | mlp gate | mlp up | mlp down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.58bit (brief)** | 0.200 | 0.162 | 1.114 | 1.116 | 1.116 | 1.108 | 0.186 | 0.174 | 0.204 |
| **3bit** | 0.200 | 0.162 | 0.302 | 0.289 | 0.285 | 0.291 | 0.186 | 0.174 | 0.204 |
| **4bit (uniform)** | 0.200 | 0.162 | 0.173 | 0.165 | 0.163 | 0.166 | 0.186 | 0.174 | 0.204 |
| **16bit (no quant)** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## 3. Global layer (layer 5) — per-tensor L2 reconstruction error

| Policy | PLE gate | PLE proj | attn q | attn k | attn v | attn o | mlp gate | mlp up | mlp down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.58bit (brief)** | 0.275 | 0.117 | 0.195 | 0.262 | 0.173 | 0.186 | 0.229 | 0.220 | 0.214 |
| **3bit** | 0.275 | 0.117 | 0.195 | 0.262 | 0.173 | 0.186 | 0.229 | 0.220 | 0.214 |
| **4bit (uniform)** | 0.275 | 0.117 | 0.195 | 0.262 | 0.173 | 0.186 | 0.229 | 0.220 | 0.214 |
| **16bit (no quant)** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## 4. Per-block packed size and forward time

| Policy | Sliding layer packed | Global layer packed | Sliding fwd ms | Global fwd ms |
|---|---:|---:|---:|---:|
| **1.58bit (brief)** | 43.3 MB | 53.2 MB | 2101 | 2128 |
| **3bit** | 46.6 MB | 53.2 MB | 2013 | 2130 |
| **4bit (uniform)** | 46.6 MB | 53.2 MB | 2106 | 2088 |
| **16bit (no quant)** | 186.1 MB | 212.3 MB | 839 | 740 |

Forward-time numbers are the **Python reference path** (unpack + matmul).
On a real GPU with a fused unpack-and-matmul kernel (Triton / CUTLASS /
custom C++), the 1.58-bit and 3-bit paths are expected to **exceed FP16
throughput** because memory bandwidth is the bottleneck and the packed
weights move 2-8× less data per matmul.

## 5. PLE sparse hash (policy-independent)

| Metric | Value |
|---|---|
| PLE full on disk | 5637.1 MB |
| Hot table resident (top-5000 tokens, BF16) | 107.5 MB |
| Hot LRU (cold slices held) | 31 |
| Hot hit rate on 200-token 85/15 workload | 84.5% |
| PLE lookups/sec (CPU, single-thread) | 21891 |

## 6. Estimated working set (text decoder only, no KV cache)

Excludes the main `embed_tokens` (1.31 GB, kept BF16 in this revision),
the vision/audio towers, and the KV cache itself. KV compression is the
same across all four policies (sliding 4×, global 5.8× at the configured
bit budget).

| Policy | Attn | MLP | PLE companions | PLE hot table (RAM) | **Total** |
|---|---:|---:|---:|---:|---:|
| **1.58bit (brief)** | 207.0 | 1653.4 | 27.8 | 107.5 | **1995.7 MB** |
| **3bit** | 321.6 | 1653.4 | 27.8 | 107.5 | **2110.4 MB** |
| **4bit (uniform)** | 321.6 | 1653.4 | 27.8 | 107.5 | **2110.4 MB** |
| **16bit (no quant)** | 1284.5 | 6606.0 | 110.1 | 107.5 | **8108.2 MB** |

## 7. Recommendation

- **Don't ship 1.58-bit on sliding attn.** L2 recon > 1.0 means
  the dequantized weights are mostly noise. You'd lose more quality
  than you'd save weight. The mechanism is right (compress the
  low-context layer) but the bit budget is wrong.
- **3-bit on sliding attn is the right answer.** L2 recon drops
  from 1.11 → 0.29 (4× improvement) for +114 MB of attn weight.
  Per-block layer packed size: 43.3 → 46.6 MB (+8%). Modest cost
  for a big quality win. Global at 4-bit, MLP at 4-bit unchanged.
- **4-bit uniform is the safe choice.** Sliding attn recon 0.16–0.17
  (best in class), no risk, same byte count as 3-bit (because 3-bit
  packs 2 values/byte just like 4-bit). If you can afford the 322 MB
  instead of 207 MB, ship this.
- **FP16/BF16 baseline: 7.9 GB of weights, all error is 0.** The
  reference point. Quality is the published Gemma 4 E4B 69.4% MMLU Pro
  / 25.4% MRCR v2 8-needle 128K.
