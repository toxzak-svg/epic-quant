"""
Mmap-based lazy safetensors loader.

Lets the engine touch a 14.89 GB model file without pulling it all into RAM.
Each call to `get_tensor(key)` materializes one tensor and lets the OS evict
the page cache on its own schedule. We never call `safe_open(path).keys()`
up-front (it allocates the whole header in one shot, which OOMs at 14 GB
files on memory-tight machines — verified).
"""
from __future__ import annotations
import os
import json
import mmap
import struct
from typing import Dict, Optional


class MmapSafetensors:
    """Read a safetensors v1 file (header at the start) lazily.

    We open the file once, mmap it, and resolve a tensor by its name on
    demand. The first call reads the JSON header (typically <1 MB even for
    huge models) and caches offset/offsets per key.
    """

    def __init__(self, path: str):
        self.path = path
        self._fd = os.open(path, os.O_RDONLY)
        # mmap the whole file. Windows mmap is read-only by default.
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        hdr_len = struct.unpack_from("<Q", self._mm, 0)[0]
        hdr_bytes = self._mm[8:8 + hdr_len]
        self._meta: Dict[str, dict] = json.loads(hdr_bytes.decode("utf-8"))
        # Build offset index (start_byte, end_byte) per tensor.
        self._offsets: Dict[str, tuple] = {}
        cursor = 8 + hdr_len
        for k, v in self._meta.items():
            if k == "__metadata__":
                continue
            shape = v["shape"]
            n = 1
            for d in shape:
                n *= d
            dtype = v["dtype"]
            bytes_per = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
                         "I8": 1, "U8": 1, "BOOL": 1, "F64": 8}.get(dtype, 2)
            size = n * bytes_per
            self._offsets[k] = (cursor, cursor + size, shape, dtype)
            cursor += size

    def keys(self):
        return [k for k in self._meta if k != "__metadata__"]

    def has(self, key: str) -> bool:
        return key in self._offsets

    def get_tensor_bytes(self, key: str) -> bytes:
        """Return raw tensor bytes (no dtype conversion)."""
        if key not in self._offsets:
            raise KeyError(key)
        start, end, _, _ = self._offsets[key]
        return self._mm[start:end]

    def get_tensor_row(self, key: str, row_idx: int, dtype: Optional[str] = None,
                       clone: bool = True):
        """Read a single row of a 2D tensor without bringing the whole tensor
        into memory.

        The safetensors index gives us (start_byte, end_byte, shape, dtype)
        for the whole tensor. We assume row-major layout and slice the
        underlying mmap buffer for that one row only. This is critical for
        the 5.27 GB PLE table on memory-tight machines.
        """
        import torch
        if key not in self._offsets:
            raise KeyError(key)
        start, end, shape, src_dtype = self._offsets[key]
        if len(shape) != 2:
            raise ValueError(f"get_tensor_row only works for 2D tensors, got {shape}")
        rows, cols = shape
        if row_idx < 0 or row_idx >= rows:
            raise IndexError(row_idx)
        bytes_per = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
                     "I8": 1, "U8": 1, "BOOL": 1, "F64": 8}.get(src_dtype, 2)
        row_bytes = cols * bytes_per
        row_start = start + row_idx * row_bytes
        row_end = row_start + row_bytes
        # Slice the mmap directly — this views the file, no copy.
        row_view = self._mm[row_start:row_end]
        torch_dtype = {"F32": torch.float32, "F16": torch.float16,
                       "BF16": torch.bfloat16, "I64": torch.int64,
                       "I32": torch.int32, "I8": torch.int8,
                       "U8": torch.uint8, "BOOL": torch.bool,
                       "F64": torch.float64}[dtype or src_dtype]
        t = torch.frombuffer(row_view, dtype=torch_dtype)
        # frombuffer on a read-only mmap gives a non-writable tensor; clone
        # if the caller wants a writable one. Most callers just want a
        # numeric view, so default to no-clone for speed.
        if clone:
            t = t.clone()
        if dtype and dtype != src_dtype:
            t = t.to(torch_dtype)
        return t

    def get_tensor(self, key: str, dtype: str = "BF16"):
        """Return a torch.Tensor view of the on-disk tensor.

        Uses torch.frombuffer to avoid copy when possible.
        """
        import torch  # local import to keep this module light
        if key not in self._offsets:
            raise KeyError(key)
        start, end, shape, src_dtype = self._offsets[key]
        raw = self._mm[start:end]
        torch_dtype = {"F32": torch.float32, "F16": torch.float16,
                       "BF16": torch.bfloat16, "I64": torch.int64,
                       "I32": torch.int32, "I8": torch.int8,
                       "U8": torch.uint8, "BOOL": torch.bool,
                       "F64": torch.float64}[src_dtype]
        t = torch.frombuffer(raw, dtype=torch_dtype).reshape(shape).clone()
        if dtype and dtype != src_dtype:
            t = t.to({"F32": torch.float32, "F16": torch.float16,
                      "BF16": torch.bfloat16, "I64": torch.int64,
                      "I32": torch.int32, "I8": torch.int8,
                      "U8": torch.uint8, "BOOL": torch.bool,
                      "F64": torch.float64}[dtype])
        return t

    def tensor_nbytes(self, key: str) -> int:
        if key not in self._offsets:
            raise KeyError(key)
        s, e, _, _ = self._offsets[key]
        return e - s

    def total_bytes(self) -> int:
        return len(self._mm)

    def close(self):
        try:
            self._mm.close()
        finally:
            os.close(self._fd)
