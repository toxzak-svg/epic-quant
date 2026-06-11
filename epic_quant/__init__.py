"""
EPIC-Quant: Epi-Stochastic Predictive Fetching & Context-Aware Bit-Shifting
for Gemma 4 E4B.

Implements the three pillars that the actual E4B architecture supports:
  1. Per-Layer Embedding (PLE) sparse hash — vocab-tier caching.
  2. Layer-type-aware weight quantization (sliding 2-bit, global 4-bit).
  3. p-RoPE frequency-aware KV cache eviction.

Drops from the original brief (with reasons):
  - "Epi-Stochastic Fetching" of expert slices — E4B is dense, not MoE.
  - "Native MTP drafter" prefetch — not present in this model's config.
  - "E4B is multimodal so MTP must be there" — false; E4B's text tower is
    dense 42-layer, no MTP head in the safetensors.

Real model shapes (verified against /root/.lmstudio/.../model.safetensors,
7.996 B params total, 2130 tensors, BF16):

  text layers 0..41, layer_types in 5+1 pattern (5 sliding_attention then
  1 full_attention, repeated 7x = 42 layers, 7 of which are full_attention).
  Sliding: head_dim=256,  q_proj[2048,2560], k/v_proj[512,2560], o[2560,2048]
  Global:  head_dim=512,  q_proj[4096,2560], k/v_proj[1024,2560], o[2560,4096]
  MLP all layers: gate[10240,2560], up[10240,2560], down[2560,10240]
  Norms: input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
         post_feedforward_layernorm, post_per_layer_input_norm  (all [2560])
  PLE:    embed_tokens_per_layer[262144, 10752]  (single 2D matrix,
         columns = 42 layers x 256 per-layer hidden)
         per_layer_input_gate[256, 2560]  per layer
         per_layer_projection[2560, 256]  per layer
         layer_scalar[1]                  per layer
  Head:   tied with embed_tokens[262144, 2560] (no separate lm_head)
  Sliding window = 512 tokens.
  p-RoPE on global: partial_rotary_factor=0.25, rope_theta=1e6.
  num_kv_shared_layers=18 (K/V are RECOMPUTED per layer; shared-KV is a
    runtime feature where the K/V from one layer is reused for the next,
    not a parameter aliasing).
"""
from .engine import EPICQuantEngine, QuantPolicy, PLEPolicy, KVPolicy
from .loader import MmapSafetensors

__all__ = [
    "EPICQuantEngine", "QuantPolicy", "PLEPolicy", "KVPolicy",
    "MmapSafetensors",
]
