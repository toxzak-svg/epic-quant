"""
Layer-aware parameter key layout for Gemma 4 E4B.

Verified directly against the safetensors header (2130 tensors).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LayerDims:
    """Dimensions for a single layer, variant-aware."""
    layer_idx: int
    is_global: bool            # full_attention vs sliding_attention
    head_dim: int              # 256 sliding, 512 global
    q_out: int                 # 2048 sliding, 4096 global
    kv_out: int                # 512 sliding, 1024 global
    hidden: int                # 2560 always


def layer_param_keys(layer_idx: int) -> List[str]:
    """The 17 BF16 parameter keys that make up one language-model decoder block.

    Plus the shared PLE contribution that this layer reads from the *single*
    2D embed_tokens_per_layer table.
    """
    base = f"model.language_model.layers.{layer_idx}"
    return [
        f"{base}.input_layernorm.weight",
        f"{base}.post_attention_layernorm.weight",
        f"{base}.pre_feedforward_layernorm.weight",
        f"{base}.post_feedforward_layernorm.weight",
        f"{base}.post_per_layer_input_norm.weight",
        f"{base}.self_attn.q_proj.weight",
        f"{base}.self_attn.k_proj.weight",
        f"{base}.self_attn.v_proj.weight",
        f"{base}.self_attn.o_proj.weight",
        f"{base}.self_attn.q_norm.weight",
        f"{base}.self_attn.k_norm.weight",
        f"{base}.mlp.gate_proj.weight",
        f"{base}.mlp.up_proj.weight",
        f"{base}.mlp.down_proj.weight",
        f"{base}.per_layer_input_gate.weight",
        f"{base}.per_layer_projection.weight",
        f"{base}.layer_scalar",
    ]


def ple_columns_for_layer(layer_idx: int, num_layers: int = 42,
                          per_layer_dim: int = 256) -> Tuple[int, int]:
    """PLE is a single 2D matrix [vocab, num_layers * per_layer_dim].

    Each layer's slice is columns [layer_idx * per_layer_dim,
    (layer_idx + 1) * per_layer_dim)."""
    start = layer_idx * per_layer_dim
    end = start + per_layer_dim
    return start, end


def get_layer_dims(layer_idx: int, layer_types: List[str]) -> LayerDims:
    is_global = layer_types[layer_idx] == "full_attention"
    if is_global:
        return LayerDims(layer_idx=layer_idx, is_global=True,
                         head_dim=512, q_out=4096, kv_out=1024, hidden=2560)
    return LayerDims(layer_idx=layer_idx, is_global=False,
                     head_dim=256, q_out=2048, kv_out=512, hidden=2560)
