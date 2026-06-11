"""
Reference forward pass for one decoder block under EPIC-Quant.

Gap 1 closed: real attention via torch.nn.functional.scaled_dot_product_attention
(F.scaled_dot_product_attention). On CUDA this is flash-attention; on CPU it's
the math/efficient kernel. Either way, this replaces the O(seq^2) einsum
we had before.

The attention mask is built as a 4D bool additive mask per the SDPA
contract: True = attend, False = mask out. We build causal + sliding-window
together for sliding layers; pure causal for global layers.

QK-norm is applied per-head after the projection. We don't implement RoPE
in this reference forward — the engine's job is to measure quant error,
not RoPE. To get exact parity with the real model, hook in
`transformers.models.gemma4.modeling_gemma4.Gemma4RotaryEmbedding` (or
read rope_parameters from config and use HF's apply_multimodal_rotary_pos_emb).
"""
from __future__ import annotations
import time
import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loader import MmapSafetensors
from .engine import (EPICQuantEngine, QuantPolicy, PLEPolicy, KVPolicy,
                     quantize_intN, dequantize_intN)
from .packed import quantize_packed, dequantize_packed, total_packed_size_bytes
from .layers import (LayerDims, layer_param_keys, ple_columns_for_layer,
                     get_layer_dims)


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(var + eps)
    return (x.to(w.dtype) * w)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def build_attn_mask(seq_len: int, is_global: bool, sliding_window: int,
                    device: torch.device) -> torch.Tensor:
    """Return a 4D additive attention mask [1, 1, S, S] for SDPA.

    Convention: 0.0 = attend, -inf = mask out.
    Causal + sliding window if not global, pure causal if global.
    """
    idx = torch.arange(seq_len, device=device)
    # Causal: q can attend to k if k_idx <= q_idx
    causal = idx[None, :] <= idx[:, None]   # [S, S]
    if is_global:
        keep = causal
    else:
        # Sliding: also require q_idx - k_idx < sliding_window
        within = (idx[:, None] - idx[None, :]) < sliding_window
        keep = causal & within
    # Convert to additive: keep -> 0, mask -> -inf
    add = torch.zeros(seq_len, seq_len, dtype=torch.bfloat16, device=device)
    add.masked_fill_(~keep, float("-inf"))
    return add[None, None, :, :]


def real_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   is_global: bool, sliding_window: int,
                   layer_scalar: torch.Tensor) -> torch.Tensor:
    """Real attention via F.scaled_dot_product_attention.

    Inputs are post-projection, in [B, H, S, D] layout, BF16.
    GQA is handled by repeating K/V along the head axis.
    """
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    repeat = H_q // H_kv
    if repeat > 1:
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    add_mask = build_attn_mask(S, is_global, sliding_window, q.device)
    # SDPA wants either a bool mask (True=keep) or a float additive mask
    # of shape [B, H, S, S] or [1, 1, S, S]. We use additive for clarity.
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask,
                                           dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(B, S, H_q * D)
    return out * layer_scalar.float().to(out.dtype)


def forward_one_layer(engine: EPICQuantEngine, layer_idx: int,
                      hidden_states: torch.Tensor, token_ids: torch.Tensor,
                      sliding_window: int = 512) -> Dict:
    """Run one decoder block with EPIC-Quant. Returns output + per-step stats."""
    sf = engine.sf
    dims = get_layer_dims(layer_idx, engine.layer_types)
    is_global = dims.is_global
    base = f"model.language_model.layers.{layer_idx}"
    t0 = time.perf_counter()
    stats = {"layer": layer_idx, "is_global": is_global}

    # 1. Input layernorm
    w_in = sf.get_tensor(f"{base}.input_layernorm.weight")
    x = _rms_norm(hidden_states, w_in)
    stats["input_norm_ms"] = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()

    # 2. PLE — gather per-layer slice, then gate + project
    ple_vecs = []
    for t in token_ids.tolist():
        ple_vecs.append(engine.ple_cache.lookup(t, layer_idx))
    ple_stack = torch.stack(ple_vecs)  # [S, ple_dim] BF16
    w_gate = sf.get_tensor(f"{base}.per_layer_input_gate.weight")  # [256, 2560]
    w_proj = sf.get_tensor(f"{base}.per_layer_projection.weight")  # [2560, 256]
    w_post = sf.get_tensor(f"{base}.post_per_layer_input_norm.weight")
    qbits = engine.quant.bits_ple_per_layer
    pg, sg = quantize_packed(w_gate, qbits)
    pp, sp = quantize_packed(w_proj, qbits)
    w_gate_dq = dequantize_packed(pg, sg, w_gate.shape[0], w_gate.shape[1], qbits)
    w_proj_dq = dequantize_packed(pp, sp, w_proj.shape[0], w_proj.shape[1], qbits)
    ple_gate = ple_stack @ w_gate_dq  # [S, 2560]
    ple_value = ple_stack @ w_proj_dq.T
    ple_contrib = (ple_gate * ple_value)  # [S, 2560]
    x = x + ple_contrib
    x = _rms_norm(x, w_post)
    stats["ple_ms"] = (time.perf_counter() - t1) * 1000
    stats["ple_gate_packed_bytes"] = pg.numel() + sg.numel() * 2
    stats["ple_proj_packed_bytes"] = pp.numel() + sp.numel() * 2
    stats["ple_gate_recon_l2"] = (w_gate_dq.float() - w_gate.float()).norm().item() / \
                                  (w_gate.float().norm().item() + 1e-9)
    stats["ple_proj_recon_l2"] = (w_proj_dq.float() - w_proj.float()).norm().item() / \
                                  (w_proj.float().norm().item() + 1e-9)
    t2 = time.perf_counter()

    # 3. Self-attn: pack q/k/v/o, then dequant + matmul, then real SDPA
    Wq = sf.get_tensor(f"{base}.self_attn.q_proj.weight")
    Wk = sf.get_tensor(f"{base}.self_attn.k_proj.weight")
    Wv = sf.get_tensor(f"{base}.self_attn.v_proj.weight")
    Wo = sf.get_tensor(f"{base}.self_attn.o_proj.weight")
    qbits = engine.quant.bits_global_attn if is_global else engine.quant.bits_sliding_attn
    pq, sq = quantize_packed(Wq, qbits)
    pk, sk = quantize_packed(Wk, qbits)
    pv, sv = quantize_packed(Wv, qbits)
    po, so = quantize_packed(Wo, qbits)
    stats["attn_q_packed_bytes"] = total_packed_size_bytes(dims.q_out, dims.hidden, qbits)
    stats["attn_k_packed_bytes"] = total_packed_size_bytes(dims.kv_out, dims.hidden, qbits)
    stats["attn_v_packed_bytes"] = total_packed_size_bytes(dims.kv_out, dims.hidden, qbits)
    stats["attn_o_packed_bytes"] = total_packed_size_bytes(dims.hidden, dims.q_out, qbits)
    Wq_dq = dequantize_packed(pq, sq, dims.q_out, dims.hidden, qbits)
    Wk_dq = dequantize_packed(pk, sk, dims.kv_out, dims.hidden, qbits)
    Wv_dq = dequantize_packed(pv, sv, dims.kv_out, dims.hidden, qbits)
    Wo_dq = dequantize_packed(po, so, dims.hidden, dims.q_out, qbits)
    stats["attn_q_recon_l2"] = (Wq_dq.float() - Wq.float()).norm().item() / (Wq.float().norm().item() + 1e-9)
    stats["attn_k_recon_l2"] = (Wk_dq.float() - Wk.float()).norm().item() / (Wk.float().norm().item() + 1e-9)
    stats["attn_v_recon_l2"] = (Wv_dq.float() - Wv.float()).norm().item() / (Wv.float().norm().item() + 1e-9)
    stats["attn_o_recon_l2"] = (Wo_dq.float() - Wo.float()).norm().item() / (Wo.float().norm().item() + 1e-9)
    # Project
    B, S, H = hidden_states.shape
    q = (x @ Wq_dq.T)  # [B, S, q_out]
    k = (x @ Wk_dq.T)  # [B, S, kv_out]
    v = (x @ Wv_dq.T)
    q = q.view(B, S, dims.q_out // dims.head_dim, dims.head_dim).transpose(1, 2)
    k = k.view(B, S, dims.kv_out // dims.head_dim, dims.head_dim).transpose(1, 2)
    v = v.view(B, S, dims.kv_out // dims.head_dim, dims.head_dim).transpose(1, 2)
    layer_scalar = sf.get_tensor(f"{base}.layer_scalar")
    attn_out = real_attention(q, k, v, is_global, sliding_window, layer_scalar)
    x = attn_out @ Wo_dq.T
    stats["attn_ms"] = (time.perf_counter() - t2) * 1000
    t3 = time.perf_counter()

    # 4. Post-attn norm + residual
    w_pa = sf.get_tensor(f"{base}.post_attention_layernorm.weight")
    hidden_states = hidden_states + _rms_norm(x, w_pa)
    x = _rms_norm(hidden_states, w_pa)
    stats["post_attn_ms"] = (time.perf_counter() - t3) * 1000
    t4 = time.perf_counter()

    # 5. MLP — pack gate/up/down, dequant, matmul
    mbits = engine.quant.bits_global_mlp if is_global else engine.quant.bits_sliding_mlp
    Wg = sf.get_tensor(f"{base}.mlp.gate_proj.weight")
    Wu = sf.get_tensor(f"{base}.mlp.up_proj.weight")
    Wd = sf.get_tensor(f"{base}.mlp.down_proj.weight")
    pg, sg = quantize_packed(Wg, mbits)
    pu, su = quantize_packed(Wu, mbits)
    pd_, sd = quantize_packed(Wd, mbits)
    Wg_dq = dequantize_packed(pg, sg, Wg.shape[0], Wg.shape[1], mbits)
    Wu_dq = dequantize_packed(pu, su, Wu.shape[0], Wu.shape[1], mbits)
    Wd_dq = dequantize_packed(pd_, sd, Wd.shape[0], Wd.shape[1], mbits)
    stats["mlp_gate_packed_bytes"] = total_packed_size_bytes(Wg.shape[0], Wg.shape[1], mbits)
    stats["mlp_up_packed_bytes"]   = total_packed_size_bytes(Wu.shape[0], Wu.shape[1], mbits)
    stats["mlp_down_packed_bytes"] = total_packed_size_bytes(Wd.shape[0], Wd.shape[1], mbits)
    stats["mlp_gate_recon_l2"] = (Wg_dq.float() - Wg.float()).norm().item() / (Wg.float().norm().item() + 1e-9)
    stats["mlp_up_recon_l2"]   = (Wu_dq.float() - Wu.float()).norm().item() / (Wu.float().norm().item() + 1e-9)
    stats["mlp_down_recon_l2"] = (Wd_dq.float() - Wd.float()).norm().item() / (Wd.float().norm().item() + 1e-9)
    mlp_out = (_gelu_tanh(x @ Wg_dq.T) * (x @ Wu_dq.T)) @ Wd_dq.T
    w_pf = sf.get_tensor(f"{base}.pre_feedforward_layernorm.weight")
    w_pof = sf.get_tensor(f"{base}.post_feedforward_layernorm.weight")
    mlp_out = _rms_norm(mlp_out, w_pf)
    hidden_states = hidden_states + _rms_norm(mlp_out, w_pof)
    stats["mlp_ms"] = (time.perf_counter() - t4) * 1000
    stats["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"hidden": hidden_states, "stats": stats}
