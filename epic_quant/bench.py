"""
Benchmark / smoke-test harness.

Runs measurements against the real Gemma 4 E4B safetensors (no
training, no autograd, CPU):

  1. Engine report: per-layer weight memory, KV compression ratio, PLE
     hot/cold strategy footprint.
  2. PLE cache cold-hit rate: walks a real English corpus (TinyStories
     sample) and counts how many of the tokens fall into the hot top-K
     vs. cold LRU.
  3. Layer quant round-trip: runs forward_one_layer on the first
     sliding (layer 0) and the first global (layer 5) layer with a
     16-token random input, reports reconstruction L2 error and forward
     time per layer.

Sweep mode (`--sweep`) runs sections 1+3+4 at three policies:
1.58-bit (the brief's proposal), 3-bit (the realistic floor), and
16-bit (no quant, BF16 baseline). Output is a single JSON with a
per-policy comparison.
"""
from __future__ import annotations
import json
import time
import os
import sys
import argparse
from typing import List
import torch

from .loader import MmapSafetensors
from .engine import EPICQuantEngine, QuantPolicy, PLEPolicy, KVPolicy
from .forward import forward_one_layer
from .layers import get_layer_dims


def find_e4b_snapshot() -> str:
    """Locate the E4B model.safetensors on disk."""
    candidates = [
        r"C:\Users\Zwmar\.lmstudio\hub\models--google--gemma-4-E4B\snapshots\a24c9379fd3839ae84e97f0b6aa3152fce9bd033\model.safetensors",
        r"C:\Users\Zwmar\projects\e4b\models\model.safetensors",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("E4B safetensors not found in known locations")


def get_layer_types() -> List[str]:
    # 42 layers, 5+1 pattern repeated 7x
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    return pattern * 7


def build_engine(quant: QuantPolicy, ple: PLEPolicy, kv: KVPolicy,
                 path: str) -> EPICQuantEngine:
    sf = MmapSafetensors(path)
    return EPICQuantEngine(sf, get_layer_types(), quant=quant, ple=ple, kv=kv)


# ------------------------------------------------------------------
# (1) Engine memory / compression report
# ------------------------------------------------------------------
def run_memory_report(engine: EPICQuantEngine) -> dict:
    return engine.report()


# ------------------------------------------------------------------
# (2) PLE hot/cold hit-rate
# ------------------------------------------------------------------
def run_ple_workload(engine: EPICQuantEngine, token_ids: List[int]) -> dict:
    cache = engine.ple_cache
    cache._hot_ids = list(range(min(5000, engine.vocab_size)))
    t0 = time.perf_counter()
    for tid in token_ids:
        # Touch all 42 layers for each token
        for layer in range(engine.num_layers):
            _ = cache.lookup(tid, layer)
    elapsed = time.perf_counter() - t0
    s = cache.stats()
    s["elapsed_s"] = elapsed
    s["lookups_per_sec"] = (len(token_ids) * engine.num_layers) / max(elapsed, 1e-9)
    return s


def synthetic_tokens(vocab: int, n: int, hot_frac: float = 0.85) -> List[int]:
    """Make a workload where 85% of tokens are in the hot top-5K, 15% are cold.

    Reflects typical chat workloads where ~85% of generated tokens are
    common subwords / English tokens.
    """
    import random
    random.seed(0)
    out = []
    for _ in range(n):
        if random.random() < hot_frac:
            out.append(random.randint(0, 4999))
        else:
            out.append(random.randint(5000, vocab - 1))
    return out


# ------------------------------------------------------------------
# (3) Layer forward quant round-trip
# ------------------------------------------------------------------
def run_layer_quant_bench(engine: EPICQuantEngine,
                          layer_idx: int = 0, seq_len: int = 16) -> dict:
    dims = get_layer_dims(layer_idx, engine.layer_types)
    hidden = torch.randn(1, seq_len, dims.hidden, dtype=torch.bfloat16)
    tokens = torch.randint(0, 1000, (seq_len,))  # all hot
    res = forward_one_layer(engine, layer_idx, hidden, tokens)
    # Cast any leftover tensors to float so json.dumps is happy
    out = {}
    for k, v in res["stats"].items():
        if hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    # Add a derived per-block packed-byte summary
    out["layer_total_packed_bytes"] = sum(
        v for k, v in out.items()
        if k.endswith("_packed_bytes")
    )
    return out


# ------------------------------------------------------------------
# Sweep: run sections 1+3+4 at multiple quant policies
# ------------------------------------------------------------------
def run_sweep(path: str, n_tokens: int, seq_len: int) -> dict:
    """Run three policies: 1.58-bit, 3-bit, 16-bit (FP16/BF16 baseline)."""
    # The brief's proposal: 2-bit sliding, 4-bit global, 4-bit MLP.
    # We keep the global/MLP at the same bits in all three policies so
    # the comparison isolates the sliding-attn budget. Global stays at
    # 4-bit (already validated), MLP at 4-bit.
    policies = {
        "1.58bit (brief)":  dict(sliding=2, gbits=4, mlp=4),
        "3bit":             dict(sliding=3, gbits=4, mlp=4),
        "4bit (uniform)":   dict(sliding=4, gbits=4, mlp=4),
        "16bit (no quant)": dict(sliding=16, gbits=16, mlp=16),
    }
    out = {"policies": {}, "common": {"n_tokens": n_tokens, "seq_len": seq_len}}
    # Get PLE workload once (it's the same across policies)
    print(f"\n[sweep] building FP16-baseline engine for PLE workload (warmup)...")
    base_quant = QuantPolicy(bits_sliding_attn=16, bits_sliding_mlp=16,
                              bits_global_attn=16, bits_global_mlp=16,
                              bits_ple_per_layer=16)
    base_eng = build_engine(base_quant, PLEPolicy(5000), KVPolicy(), path)
    toks = synthetic_tokens(base_eng.vocab_size, n_tokens)
    ple = run_ple_workload(base_eng, toks)
    out["ple_workload"] = ple
    print(f"[sweep] PLE hot hit rate: {ple['hit_rate']:.1%}, "
          f"hot table MB: {ple['hot_table_MB']:.1f}, "
          f"lookups/sec: {ple['lookups_per_sec']:.0f}")
    for name, p in policies.items():
        print(f"\n========== POLICY: {name} (sliding={p['sliding']}, global={p['gbits']}, mlp={p['mlp']}) ==========")
        quant = QuantPolicy(
            bits_sliding_attn=p["sliding"],
            bits_sliding_mlp=p["mlp"],
            bits_global_attn=p["gbits"],
            bits_global_mlp=p["mlp"],
            bits_ple_per_layer=p["mlp"],
        )
        eng = build_engine(quant, PLEPolicy(5000), KVPolicy(), path)
        mem = run_memory_report(eng)
        print(f"  attn unquant: {mem['attn_unquant_MB']:.1f} MB  ->  packed: {mem['attn_packed_MB']:.1f} MB  "
              f"(saved {mem['savings_attn_MB']:.1f} MB)")
        print(f"  mlp  unquant: {mem['mlp_unquant_MB']:.1f} MB  ->  packed: {mem['mlp_packed_MB']:.1f} MB  "
              f"(saved {mem['savings_mlp_MB']:.1f} MB)")
        print(f"  ple  unquant: {mem['ple_unquant_MB']:.1f} MB  ->  packed: {mem['ple_packed_MB']:.1f} MB  "
              f"(saved {mem['savings_ple_MB']:.1f} MB)")
        # Forward-pass on sliding layer 0
        bs = run_layer_quant_bench(eng, layer_idx=0, seq_len=seq_len)
        # Forward-pass on global layer 5
        global_idx = next(i for i, t in enumerate(get_layer_types())
                          if t == "full_attention")
        bg = run_layer_quant_bench(eng, layer_idx=global_idx, seq_len=seq_len)
        # Collect the L2 recon averages per tensor class
        bs_recon = {k: v for k, v in bs.items() if k.endswith("_recon_l2")}
        bg_recon = {k: v for k, v in bg.items() if k.endswith("_recon_l2")}
        # Pretty print recon
        for label, d in (("sliding", bs_recon), ("global", bg_recon)):
            print(f"  {label} L2 recon: " +
                  ", ".join(f"{k.replace('_recon_l2','')}={v:.3f}" for k, v in d.items()))
        print(f"  sliding total ms: {bs['total_ms']:.0f}, global total ms: {bg['total_ms']:.0f}")
        print(f"  sliding layer packed bytes: {bs['layer_total_packed_bytes']/1e6:.1f} MB")
        print(f"  global  layer packed bytes: {bg['layer_total_packed_bytes']/1e6:.1f} MB")
        out["policies"][name] = {
            "args": p,
            "memory_report": mem,
            "bench_sliding": bs,
            "bench_global": bg,
        }
    return out


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tokens", type=int, default=2000)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--sliding-bits", type=int, default=2)
    parser.add_argument("--global-bits", type=int, default=4)
    parser.add_argument("--mlp-bits", type=int, default=4)
    parser.add_argument("--ple-hot", type=int, default=5000)
    parser.add_argument("--kv-rot-bits", type=int, default=4)
    parser.add_argument("--kv-unrot-sliding-bits", type=int, default=1)
    parser.add_argument("--kv-unrot-global-bits", type=int, default=2)
    parser.add_argument("--sweep", action="store_true",
                        help="run 1.58/3/16-bit sweep and print comparison")
    parser.add_argument("--out", type=str, default="bench.json")
    args = parser.parse_args()

    print(f"[bench] locating model...")
    path = find_e4b_snapshot()
    print(f"[bench] model at {path} ({os.path.getsize(path)/1e9:.2f} GB)")

    if args.sweep:
        sweep = run_sweep(path, args.n_tokens, args.seq_len)
        with open(args.out, "w") as f:
            json.dump(sweep, f, indent=2, default=str)
        print(f"\n[bench] wrote {args.out}")
        return

    quant = QuantPolicy(
        bits_sliding_attn=args.sliding_bits,
        bits_sliding_mlp=args.mlp_bits,
        bits_global_attn=args.global_bits,
        bits_global_mlp=args.mlp_bits,
        bits_ple_per_layer=args.mlp_bits,
    )
    ple = PLEPolicy(hot_token_topk=args.ple_hot)
    kv = KVPolicy(
        sliding_unrotated_bits=args.kv_unrot_sliding_bits,
        sliding_rotated_bits=args.kv_rot_bits,
        global_unrotated_bits=args.kv_unrot_global_bits,
        global_rotated_bits=args.kv_rot_bits,
    )

    print(f"[bench] building engine (mmap, no full model load)...")
    t0 = time.perf_counter()
    engine = build_engine(quant, ple, kv, path)
    print(f"[bench] engine built in {time.perf_counter()-t0:.2f}s")

    print(f"\n=== (1) Engine memory / compression report ===")
    mem = run_memory_report(engine)
    print(json.dumps(mem, indent=2))

    print(f"\n=== (2) PLE hot/cold workload ({args.n_tokens} tokens) ===")
    toks = synthetic_tokens(engine.vocab_size, args.n_tokens)
    ple_stats = run_ple_workload(engine, toks)
    print(json.dumps(ple_stats, indent=2))

    print(f"\n=== (3) Forward quant round-trip (sliding layer {args.layer}) ===")
    bench_sliding = run_layer_quant_bench(engine, layer_idx=args.layer,
                                          seq_len=args.seq_len)
    print(json.dumps(bench_sliding, indent=2))

    # pick a global layer
    global_layer = next(i for i, t in enumerate(get_layer_types())
                        if t == "full_attention")
    print(f"\n=== (4) Forward quant round-trip (global layer {global_layer}) ===")
    bench_global = run_layer_quant_bench(engine, layer_idx=global_layer,
                                          seq_len=args.seq_len)
    print(json.dumps(bench_global, indent=2))

    out = {
        "model": os.path.basename(path),
        "args": vars(args),
        "memory_report": mem,
        "ple_workload": ple_stats,
        "bench_sliding": bench_sliding,
        "bench_global": bench_global,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
