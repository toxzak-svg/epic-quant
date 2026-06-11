"""
Packed 2-bit ternary, 3-bit symmetric int, and 4-bit weight formats.

Reference implementation that reports real byte counts.

Packing layout:
  - 2-bit: 4 values packed into 1 byte. Each value is 2 bits representing
    one of {-1, 0, +1, "n/a"} -> we use {0, 1, 2, 3} = {-1, 0, +1, unused}.
  - 3-bit: 3 values packed into 1 byte (1 bit unused). Symmetric int in
    range [-4, 3] (8 centroids). 2.67 bits/value effective, but the
    unused bit is real overhead — see packed_size_bytes.
  - 4-bit: 2 values packed into 1 byte. Each value is int4 in [-8, 7].
  - "fp16": no quant, weights stored as BF16 (the engine's working
    dtype). 2 bytes per value. Used as a "no quant" baseline.

scales: per-row fp16 scale, one per output row. Stored separately
from the packed weights.

Byte accounting per row of an [out, in] Linear:
  - 2-bit packed: out * ceil(in/4) bytes + out * 2 bytes (scales)
  - 3-bit packed: out * ceil(in/3) bytes + out * 2 bytes (scales)
  - 4-bit packed: out * ceil(in/2) bytes + out * 2 bytes (scales)
  - FP16 (BF16):  out * in * 2 bytes (no scales needed)
"""
from __future__ import annotations
from typing import Tuple
import torch


def quantize_packed(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight matrix and return (packed_bytes, per_row_scales).

    `packed_bytes` is a uint8 tensor of shape [out, ceil(in/pack)].
    `scales` is fp16 [out].
    """
    assert w.is_floating_point() and w.dim() == 2
    out, in_dim = w.shape
    w32 = w.to(torch.float32)
    if bits == 2:
        # Ternary {-1, 0, +1}
        max_abs = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_abs.squeeze(1)  # [out]
        w_scaled = w32 / max_abs
        q = torch.zeros_like(w_scaled, dtype=torch.int8)
        q[w_scaled >  0.33] =  1
        q[w_scaled < -0.33] = -1
        q01 = (q + 1).to(torch.int64)  # {-1,0,1} -> {0,1,2}
        pad = (4 - in_dim % 4) % 4
        if pad:
            q01 = torch.cat([q01, torch.zeros(out, pad, dtype=torch.int64)], dim=1)
        in_packed = q01.shape[1] // 4
        packed = torch.zeros(out, in_packed, dtype=torch.uint8)
        for j in range(4):
            packed |= (q01[:, j::4].to(torch.uint8) << (2 * j))
        return packed, scale.to(torch.float16)
    if bits == 3:
        # Symmetric int3 in [-4, 3] (8 centroids).
        # 3 bits/value, but 3 values don't fit in a byte (3*3=9). Standard
        # packing: 2 values per byte, 2 bits wasted. So in_dim values
        # need ceil(in_dim * 3 / 8) bytes = ceil(in_dim / 2) * 3 / 4 ...
        # actually 2 values per byte at 3 bits each = 6 bits used, 2 wasted
        # per byte. Same as ceil(in_dim * 3 / 8) = ceil(in_dim * 0.375).
        max_abs = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (max_abs / 4.0).squeeze(1)
        w_scaled = (w32 / scale.unsqueeze(1)).round().clamp(-4, 3)
        q01 = (w_scaled + 4).to(torch.int64)  # [-4,3] -> [0,7]
        # Pad in_dim to even count (so 2 values/byte works cleanly)
        pad = (2 - in_dim % 2) % 2
        if pad:
            q01 = torch.cat([q01, torch.zeros(out, pad, dtype=torch.int64)], dim=1)
        # Reshape [out, in_padded] -> [out, in_padded/2, 2] and pack each
        # pair into a byte: low nibble = vals[0], high nibble shifted = vals[1].
        in_padded = q01.shape[1]
        pairs = q01.reshape(out, in_padded // 2, 2)
        v0 = pairs[:, :, 0].to(torch.uint8) & 0x7
        v1 = pairs[:, :, 1].to(torch.uint8) & 0x7
        packed = v0 | (v1 << 3)
        return packed, scale.to(torch.float16)
    if bits == 4:
        max_abs = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (max_abs / 7.0).squeeze(1)
        w_scaled = (w32 / scale.unsqueeze(1)).round().clamp(-8, 7)
        q01 = (w_scaled + 8).to(torch.int64)
        pad = (2 - in_dim % 2) % 2
        if pad:
            q01 = torch.cat([q01, torch.zeros(out, pad, dtype=torch.int64)], dim=1)
        in_packed = q01.shape[1] // 2
        packed = torch.zeros(out, in_packed, dtype=torch.uint8)
        packed |= (q01[:, 0::2].to(torch.uint8) & 0xF)
        packed |= (q01[:, 1::2].to(torch.uint8) & 0xF) << 4
        return packed, scale.to(torch.float16)
    if bits == 16:
        # FP16 path: no quant. We return the BF16 bytes of the weight
        # (cast to BF16 for engine consistency) as a "packed" buffer of
        # 1 byte per byte, plus a scale of 1s (so the dequant path is
        # uniform across bits=2/3/4/16).
        scale = torch.ones(out, dtype=torch.float16)
        # Pack BF16 weight bytes (2 per value).
        w_bf16 = w.to(torch.bfloat16).contiguous()
        packed = w_bf16.view(torch.uint8).reshape(out, in_dim * 2).contiguous()
        return packed, scale
    raise NotImplementedError(f"bits={bits}")


def dequantize_packed(packed: torch.Tensor, scales: torch.Tensor,
                      out_dim: int, in_dim: int, bits: int) -> torch.Tensor:
    """Inverse of quantize_packed. Returns BF16 weight."""
    if bits == 2:
        out, in_packed = packed.shape
        q01 = torch.zeros(out, in_packed * 4, dtype=torch.int64)
        for j in range(4):
            q01[:, j::4] = ((packed >> (2 * j)) & 0x3).to(torch.int64)
        q01 = q01[:, :in_dim]
        q = (q01 - 1).to(torch.float32)
        return (q * scales.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
    if bits == 3:
        out, in_packed = packed.shape
        q01 = torch.zeros(out, in_packed * 3, dtype=torch.int64)
        for j in range(3):
            shift = 3 * j
            mask = 0x7 if j < 2 else 0x3
            q01[:, j::3] = ((packed >> shift) & mask).to(torch.int64)
        q01 = q01[:, :in_dim]
        q = (q01 - 4).to(torch.float32)  # back to [-4, 3]
        return (q * scales.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
    if bits == 4:
        out, in_packed = packed.shape
        q01 = torch.zeros(out, in_packed * 2, dtype=torch.int64)
        q01[:, 0::2] = (packed & 0xF).to(torch.int64)
        q01[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int64)
        q01 = q01[:, :in_dim]
        q = (q01 - 8).to(torch.float32)
        return (q * scales.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
    if bits == 16:
        # Treat the 2-bytes-per-value buffer as BF16.
        flat = packed.reshape(out_dim, in_dim * 2)
        t = flat.view(torch.uint8).reshape(out_dim, in_dim, 2).view(torch.bfloat16).reshape(out_dim, in_dim).clone()
        return t
    raise NotImplementedError(f"bits={bits}")


def packed_size_bytes(out_dim: int, in_dim: int, bits: int) -> int:
    """Real on-disk / on-RAM byte count for a packed weight tensor (no scales)."""
    if bits == 2:
        return out_dim * ((in_dim + 3) // 4)
    if bits == 3:
        return out_dim * ((in_dim + 2) // 3)
    if bits == 4:
        return out_dim * ((in_dim + 1) // 2)
    if bits == 16:
        return out_dim * in_dim * 2
    raise NotImplementedError


def total_packed_size_bytes(out_dim: int, in_dim: int, bits: int) -> int:
    """packed weights + per-row fp16 scales (no scales needed at 16)."""
    if bits == 16:
        return packed_size_bytes(out_dim, in_dim, bits)
    return packed_size_bytes(out_dim, in_dim, bits) + out_dim * 2

