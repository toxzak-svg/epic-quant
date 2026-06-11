"""
EPIC-Quant engine: per-layer-type weight quantization, PLE sparse hash,
and p-RoPE-aware KV eviction. CPU-first; GPU dispatch is a thin wrapper.

The engine is *stateless* in the sense that the policy objects are immutable
once set, and all per-call state (KV cache, hot-PLE table) is held inside
KVCache / PLECache objects that the engine creates on `setup()`.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loader import MmapSafetensors
from .layers import (LayerDims, layer_param_keys, ple_columns_for_layer,
                     get_layer_dims)


# ----------------------------- policy dataclasses ----------------------------


@dataclass
class QuantPolicy:
    """How aggressively to quantize each weight tensor type."""
    # Per-layer-type weight bit-widths. Sliding layers get the aggressive
    # treatment the brief asked for; global layers get a higher fidelity
    # budget because p-RoPE on global layers rotates only 25% of the head
    # dim, so dropping precision there costs long-range recall.
    bits_sliding_attn: int = 2     # sliding-attn q/k/v/o: aggressive
    bits_sliding_mlp:  int = 4     # sliding MLP: medium
    bits_global_attn:  int = 4     # global-attn q/k/v/o: high fidelity
    bits_global_mlp:   int = 4     # global MLP: high fidelity
    bits_ple_per_layer: int = 4    # gate/projection: high fidelity
    # Norms and scalars are never quantized (already tiny).
    # Embed_tokens (the main shared 2560-dim embed) and lm_head
    # (tied with embed_tokens) are never quantized in this revision.

    def is_int_bits(self, b: int) -> bool:
        return b in (2, 3, 4, 8, 16)


@dataclass
class PLEPolicy:
    """Per-Layer Embedding sparse-hash policy.

    PLE is a single 2D matrix [262144, 42*256] = [vocab, num_layers*ple_dim].
    Each layer reads its own 256-wide column slice per token. We keep a hot
    cache of the most-frequent tokens uncompressed in RAM, and route cold
    tokens to a fallback path that materializes their columns on demand.

    `hot_token_topk`: how many of the lowest-id (or learned-top-k) tokens
    to keep uncompressed. The brief's 5000 default is sensible; we let the
    caller measure their workload and override.

    `cold_strategy`:
      - "lazy": materialize the cold token's column slice on demand, then
                evict by LRU. Cheap, slow for repeated cold tokens.
      - "stream": same as lazy but never holds cold slices in RAM.
      - "shared_index": route cold tokens to a single base-layer slice
                (i.e. use layer 0's columns for any cold token in any
                layer). Cheap and fast, but loses per-layer specificity.
    """
    hot_token_topk: int = 5000
    cold_strategy: str = "lazy"
    lru_capacity: int = 64        # how many cold slices to hold at once


@dataclass
class KVPolicy:
    """p-RoPE-aware KV cache eviction policy.

    Sliding layers have head_dim=256 and standard RoPE (theta=1e4).
    Global layers have head_dim=512 and p-RoPE (theta=1e6,
    partial_rotary_factor=0.25) — only 25% of the head dim is rotated.

    For sliding layers, drop the un-rotated 75% of the head dim (rows
    that carry only high-frequency / position-agnostic info) to 1-bit.
    Keep the rotated 25% at 4-bit.

    For global layers, keep the rotated 25% at 4-bit (it carries the
    long-range position signal) and drop the un-rotated 75% to 2-bit
    (it carries redundant short-range context that already exists in
    sliding layers' KV).
    """
    sliding_unrotated_bits: int = 1
    sliding_rotated_bits:   int = 4
    global_unrotated_bits:  int = 2
    global_rotated_bits:    int = 4


# ----------------------------- quantization kernels ----------------------------


def quantize_intN(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row intN quantization.

    Returns (q, scales). `q` is int8 storage of N-bit values, scales is fp16.
    For bits=2: pack 4 values per byte (returned as int8 expanded, NOT packed —
    the packing kernel is a separate step in the production version; this
    reference keeps it readable).

    This is the *reference* quantizer; the brief calls it ternary on sliding
    (1.58-bit = 2 values per weight {-1, 0, +1}), so for bits=2 we constrain
    the centroid set to {-1, 0, +1} which is the standard "ternary" trick.
    """
    assert w.is_floating_point()
    orig_shape = w.shape
    w_flat = w.reshape(w.shape[0], -1).to(torch.float32)
    if bits == 2:
        # Ternary: centroids {-1, 0, +1}
        max_abs = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_abs  # one scale per row
        w_scaled = w_flat / scale
        q = torch.zeros_like(w_scaled, dtype=torch.int8)
        q[w_scaled >  0.33] =  1
        q[w_scaled < -0.33] = -1
        # 0.33 threshold = the standard 1.58-bit heuristic
        return q.reshape(orig_shape).to(torch.int8), scale.reshape(-1).to(torch.float16)
    if bits == 4:
        max_abs = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 7.0
        w_scaled = (w_flat / scale).round().clamp(-8, 7)
        return w_scaled.reshape(orig_shape).to(torch.int8), scale.reshape(-1).to(torch.float16)
    raise NotImplementedError(f"bits={bits}")


def dequantize_intN(q: torch.Tensor, scales: torch.Tensor, out_shape: Tuple[int, ...],
                    bits: int) -> torch.Tensor:
    """Inverse of quantize_intN."""
    q_flat = q.reshape(q.shape[0], -1).to(torch.float32)
    if bits == 2:
        # Ternary: restore scale
        w = q_flat * scales.to(torch.float32).reshape(-1, 1)
    elif bits == 4:
        w = q_flat * scales.to(torch.float32).reshape(-1, 1)
    else:
        raise NotImplementedError
    return w.reshape(out_shape)


# ----------------------------- PLE cache ----------------------------


class PLECache:
    """Sparse PLE cache.

    The PLE table is a single 2D matrix [vocab, num_layers*ple_dim] in BF16
    on disk. Hot tokens (lowest token IDs by default) have their full column
    slice [num_layers*ple_dim] resident in RAM as BF16. Cold tokens are
    routed via one of three strategies.
    """

    def __init__(self, sf: MmapSafetensors, vocab_size: int,
                 num_layers: int, per_layer_dim: int,
                 policy: PLEPolicy):
        self.sf = sf
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.ple_dim = per_layer_dim
        self.policy = policy
        ple_key = "model.language_model.embed_tokens_per_layer.weight"
        # ple_total_bytes: full size of the 2D PLE matrix in BF16
        self.ple_total_bytes = sf.tensor_nbytes(ple_key)
        # Hot table: [hot_token_topk, num_layers*ple_dim] in BF16
        self._hot_ids = list(range(min(policy.hot_token_topk, vocab_size)))
        # Lazy materialization: load only on first lookup so that cold-id-only
        # workloads don't pay the cost.
        self._hot_table: Optional[torch.Tensor] = None
        # LRU for cold slices (full PLE column slice per cold token)
        self._cold_lru: Dict[int, torch.Tensor] = {}
        self._cold_lru_order: List[int] = []
        # Stats
        self.hits = 0
        self.misses = 0

    def _ensure_hot(self):
        if self._hot_table is not None:
            return
        # Materialize hot rows one-by-one via mmap row reads. For the brief's
        # default 5000 tokens this is 5000 * 21,504 bytes = ~103 MB resident,
        # not the full 5.27 GB.
        K = len(self._hot_ids)
        ple_key = "model.language_model.embed_tokens_per_layer.weight"
        full_cols = self.num_layers * self.ple_dim
        rows = []
        for i in range(K):
            rows.append(self.sf.get_tensor_row(ple_key, i, clone=True))
        self._hot_table = torch.stack(rows, dim=0).contiguous()  # [K, full_cols]

    def lookup(self, token_id: int, layer_idx: int) -> torch.Tensor:
        """Return the 256-dim PLE vector for (token, layer).

        Returns a [ple_dim] BF16 tensor.
        """
        col_start, col_end = ple_columns_for_layer(layer_idx, self.num_layers,
                                                    self.ple_dim)
        if token_id in self._hot_ids:
            self._ensure_hot()
            self.hits += 1
            return self._hot_table[token_id, col_start:col_end]
        # Cold path
        self.misses += 1
        if self.policy.cold_strategy == "shared_index":
            # Use layer 0's slice for any layer — fast, lossy
            self._ensure_hot()
            return self._hot_table[token_id, 0:self.ple_dim] if token_id < len(self._hot_ids) \
                else torch.zeros(self.ple_dim, dtype=torch.bfloat16)
        # For "stream" and "lazy" we use the row-level mmap read so we
        # never bring the full 5.27 GB PLE matrix into RAM.
        if self.policy.cold_strategy == "stream":
            row = self.sf.get_tensor_row(
                "model.language_model.embed_tokens_per_layer.weight",
                token_id, clone=True)
            return row[col_start:col_end].contiguous()
        # Default "lazy": check LRU, else load
        if token_id in self._cold_lru:
            self._cold_lru_order.remove(token_id)
            self._cold_lru_order.append(token_id)
            return self._cold_lru[token_id][col_start:col_end]
        # Evict LRU
        while len(self._cold_lru) >= self.policy.lru_capacity:
            evict = self._cold_lru_order.pop(0)
            del self._cold_lru[evict]
        # Per-row mmap read — only 21,504 bytes per row.
        row = self.sf.get_tensor_row(
            "model.language_model.embed_tokens_per_layer.weight",
            token_id, clone=True)
        self._cold_lru[token_id] = row
        self._cold_lru_order.append(token_id)
        return row[col_start:col_end]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hot_table_resident": self._hot_table is not None,
            "hot_table_MB": (self._hot_table.numel() * 2 / 1e6) if self._hot_table is not None else 0,
            "ple_full_MB": self.ple_total_bytes / 1e6,
            "hot_token_topk": len(self._hot_ids),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "lru_size": len(self._cold_lru),
        }


# ----------------------------- KV cache ----------------------------


class KVEvictor:
    """p-RoPE-aware KV eviction.

    Sliding layer head_dim=256, standard RoPE (theta=1e4) — all 256 dim
    is rotated. But sliding layers only see the last 512 tokens, so
    long-range position info is useless. We keep the rotated dim at 4-bit
    and drop everything else (i.e. we don't even store it) — sliding KV
    becomes "rotated 4-bit only."

    Global layer head_dim=512, p-RoPE with partial_rotary_factor=0.25 —
    only the FIRST 25% of the head dim (indices 0..127) is rotated; the
    rest is unrotated and carries position-agnostic content. Keep the
    rotated 25% at 4-bit, drop the unrotated 75% to 2-bit.

    We implement this as a custom KV container that stores K, V tensors
    in a packed form. For the reference implementation we keep them
    un-packed (just bool/mask) and report the *theoretical* memory
    reduction; the actual packing kernel is a follow-up.
    """

    def __init__(self, policy: KVPolicy, sliding_head_dim: int = 256,
                 global_head_dim: int = 512, num_kv_heads: int = 2,
                 partial_rotary_factor: float = 0.25):
        self.policy = policy
        self.sliding_head_dim = sliding_head_dim
        self.global_head_dim = global_head_dim
        self.num_kv_heads = num_kv_heads
        self.partial_rotary_factor = partial_rotary_factor

    def bit_budget(self, is_global: bool) -> int:
        """Total bits per token per KV tensor (sum of K and V)."""
        if is_global:
            head = self.global_head_dim
            rot = int(head * self.partial_rotary_factor)
        else:
            head = self.sliding_head_dim
            rot = head  # sliding: full RoPE on all 256 dim
        unrot = head - rot
        # K and V each pay this budget
        per_kv = rot * self.policy.global_rotated_bits if is_global else \
                 rot * self.policy.sliding_rotated_bits
        per_kv += unrot * (self.policy.global_unrotated_bits if is_global else
                           self.policy.sliding_unrotated_bits)
        return per_kv * 2  # K and V

    def f16_budget(self, head_dim: int) -> int:
        """Bits per token per KV at BF16 (no quant). For comparison."""
        return head_dim * 16 * 2  # K and V, BF16

    def compression_ratio(self, is_global: bool) -> float:
        head = self.global_head_dim if is_global else self.sliding_head_dim
        return self.bit_budget(is_global) / self.f16_budget(head)


# ----------------------------- engine ----------------------------


class EPICQuantEngine:
    """Top-level engine. Holds policies and constructs the per-layer buffers.

    Stateless w.r.t. the model itself: the user passes a MmapSafetensors
    handle and a layer_types list (from config.text_config.layer_types).
    """

    def __init__(self, sf: MmapSafetensors, layer_types: List[str],
                 vocab_size: int = 262144, num_layers: int = 42,
                 ple_dim: int = 256,
                 quant: Optional[QuantPolicy] = None,
                 ple:   Optional[PLEPolicy] = None,
                 kv:    Optional[KVPolicy] = None):
        self.sf = sf
        self.layer_types = layer_types
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.quant = quant or QuantPolicy()
        self.ple_policy = ple or PLEPolicy()
        self.kv_policy = kv or KVPolicy()
        self.ple_cache = PLECache(sf, vocab_size, num_layers, ple_dim, self.ple_policy)
        self.kv_evictor = KVEvictor(self.kv_policy)

    def layer_weight_budget_bits(self, layer_idx: int) -> dict:
        """For reporting: bits used for one block's quantizable weights.

        Uses the *packed* byte accounting (packed.py), which is the real
        on-RAM cost once 2-bit / 4-bit packing is applied.
        """
        from .packed import total_packed_size_bytes
        d = get_layer_dims(layer_idx, self.layer_types)
        q_bits = self.quant.bits_global_attn if d.is_global else self.quant.bits_sliding_attn
        m_bits = self.quant.bits_global_mlp  if d.is_global else self.quant.bits_sliding_mlp
        hidden = d.hidden
        # Per-linearity packed byte counts
        q_bytes  = total_packed_size_bytes(d.q_out,  hidden, q_bits)
        k_bytes  = total_packed_size_bytes(d.kv_out, hidden, q_bits)
        v_bytes  = total_packed_size_bytes(d.kv_out, hidden, q_bits)
        o_bytes  = total_packed_size_bytes(hidden,   d.q_out, q_bits)
        attn_packed = q_bytes + k_bytes + v_bytes + o_bytes
        attn_unquant = (d.q_out * hidden + 2 * d.kv_out * hidden + hidden * d.q_out) * 2
        # MLP
        gate_bytes = total_packed_size_bytes(10240, hidden, m_bits)
        up_bytes   = total_packed_size_bytes(10240, hidden, m_bits)
        down_bytes = total_packed_size_bytes(hidden, 10240, m_bits)
        mlp_packed  = gate_bytes + up_bytes + down_bytes
        mlp_unquant = (2 * 10240 * hidden + hidden * 10240) * 2
        # PLE companions (gate, projection) — kept at 4-bit
        p_bits = self.quant.bits_ple_per_layer
        ple_gate = total_packed_size_bytes(256, hidden, p_bits)  # [ple_dim, hidden]
        ple_proj = total_packed_size_bytes(hidden, 256, p_bits)  # [hidden, ple_dim]
        ple_unquant = (256 * hidden + hidden * 256) * 2
        return {
            "layer": layer_idx, "is_global": d.is_global,
            "q_bits": q_bits, "m_bits": m_bits,
            "attn_packed_MB":    attn_packed   / 1e6,
            "attn_unquant_MB":   attn_unquant  / 1e6,
            "mlp_packed_MB":     mlp_packed    / 1e6,
            "mlp_unquant_MB":    mlp_unquant   / 1e6,
            "ple_packed_MB":     (ple_gate + ple_proj) / 1e6,
            "ple_unquant_MB":    ple_unquant / 1e6,
        }

    def report(self) -> dict:
        """Per-layer memory + KV compression + PLE stats. Uses *packed* byte counts."""
        per_layer = [self.layer_weight_budget_bits(i) for i in range(self.num_layers)]
        total_attn_un = sum(p["attn_unquant_MB"] for p in per_layer)
        total_mlp_un  = sum(p["mlp_unquant_MB"]  for p in per_layer)
        total_ple_un  = sum(p["ple_unquant_MB"]  for p in per_layer)
        total_attn_p  = sum(p["attn_packed_MB"]  for p in per_layer)
        total_mlp_p   = sum(p["mlp_packed_MB"]   for p in per_layer)
        total_ple_p   = sum(p["ple_packed_MB"]   for p in per_layer)
        # KV compression
        sliding_layers = [i for i, t in enumerate(self.layer_types)
                          if t == "sliding_attention"]
        global_layers  = [i for i, t in enumerate(self.layer_types)
                          if t == "full_attention"]
        kv = {
            "sliding_compression": self.kv_evictor.compression_ratio(False),
            "global_compression":  self.kv_evictor.compression_ratio(True),
            "sliding_budget_bits_per_kv_token":
                self.kv_evictor.bit_budget(False) // 2,
            "global_budget_bits_per_kv_token":
                self.kv_evictor.bit_budget(True) // 2,
        }
        return {
            "attn_unquant_MB": total_attn_un,
            "mlp_unquant_MB":  total_mlp_un,
            "ple_unquant_MB":  total_ple_un,
            "attn_packed_MB":  total_attn_p,
            "mlp_packed_MB":   total_mlp_p,
            "ple_packed_MB":   total_ple_p,
            "savings_attn_MB": total_attn_un - total_attn_p,
            "savings_mlp_MB":  total_mlp_un  - total_mlp_p,
            "savings_ple_MB":  total_ple_un  - total_ple_p,
            "kv": kv,
            "ple": self.ple_cache.stats(),
            "n_sliding_layers": len(sliding_layers),
            "n_global_layers":  len(global_layers),
            "quant_policy": self.quant.__dict__,
            "ple_policy":   self.ple_policy.__dict__,
            "kv_policy":    self.kv_policy.__dict__,
        }
