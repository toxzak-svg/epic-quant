# -*- coding: utf-8 -*-
r"""
Local smoke test for the EPIC-Quant engine.

Chain the per-block `forward_one_layer` through all 42 layers with a
running hidden state, generate a few tokens, and report the top-1
prediction as a sanity check. Also runs at FP16 (no quant) and
3-bit, so you can eyeball whether the quantized model still produces
sensible tokens.

This runs on CPU. It is NOT a quality eval (no WikiText PPL, no MMLU
Pro). It's a smoke test: does the engine produce a coherent forward
end-to-end on real weights? See `kaggle_notebook.py` for the real
eval sweep that needs a GPU.

Usage:
    set PYTHONPATH=...project_root
    python -m epic_quant.eval.smoke --policy 3bit --prompt "The capital of France is"
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import List
import torch

from ..loader import MmapSafetensors
from ..engine import EPICQuantEngine, QuantPolicy, PLEPolicy, KVPolicy
from ..forward import forward_one_layer
from ..layers import get_layer_dims


def find_e4b_snapshot() -> str:
    candidates = [
        r"C:\Users\Zwmar\.lmstudio\hub\models--google--gemma-4-E4B\snapshots\a24c9379fd3839ae84e97f0b6aa3152fce9bd033\model.safetensors",
        r"/root/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/a24c9379fd3839ae84e97f0b6aa3152fce9bd033/model.safetensors",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("E4B safetensors not found")


def get_layer_types():
    return (["sliding_attention"] * 5 + ["full_attention"]) * 7


def policy_from_name(name: str) -> QuantPolicy:
    presets = {
        "1.58bit (brief)":  dict(sliding=2, gbits=4, mlp=4),
        "3bit":             dict(sliding=3, gbits=4, mlp=4),
        "4bit (uniform)":   dict(sliding=4, gbits=4, mlp=4),
        "16bit (no quant)": dict(sliding=16, gbits=16, mlp=16),
    }
    p = presets[name]
    return QuantPolicy(
        bits_sliding_attn=p["sliding"],
        bits_sliding_mlp=p["mlp"],
        bits_global_attn=p["gbits"],
        bits_global_mlp=p["mlp"],
        bits_ple_per_layer=p["mlp"],
    )


def run_chain(engine: EPICQuantEngine, hidden: torch.Tensor, tokens: torch.Tensor,
              layer_types: List[str]) -> torch.Tensor:
    """Chain forward_one_layer through all 42 layers. Returns final hidden.

    The current forward.py does NOT carry a real KV cache across calls;
    each call re-attends over its input. For a smoke test that's fine —
    we just need to verify the engine produces a stable output.
    """
    h = hidden
    for i in range(engine.num_layers):
        out = forward_one_layer(engine, i, h, tokens)
        h = out["hidden"]
    return h


def load_embed_and_head(sf: MmapSafetensors):
    """Load the main shared embed and (tied) lm_head weights."""
    embed = sf.get_tensor("model.language_model.embed_tokens.weight")
    # tie_word_embeddings=true -> lm_head == embed_tokens
    return embed, embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="3bit",
                        choices=["1.58bit (brief)", "3bit", "4bit (uniform)", "16bit (no quant)"])
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--out", default="smoke.json")
    args = parser.parse_args()

    print(f"[smoke] locating E4B...")
    path = find_e4b_snapshot()
    print(f"[smoke] model at {path} ({os.path.getsize(path)/1e9:.2f} GB)")

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r"C:\Users\Zwmar\.lmstudio\hub\models--google--gemma-4-E4B\snapshots\a24c9379fd3839ae84e97f0b6aa3152fce9bd033")
    encoded = tok(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]  # [1, S]
    S = input_ids.shape[1]
    print(f"[smoke] prompt: {args.prompt!r} -> {S} tokens: {input_ids[0].tolist()}")

    # Build engine
    quant = policy_from_name(args.policy)
    sf = MmapSafetensors(path)
    engine = EPICQuantEngine(sf, get_layer_types(), quant=quant,
                              ple=PLEPolicy(hot_token_topk=5000),
                              kv=KVPolicy())
    print(f"[smoke] engine built: sliding={args.policy}, num_layers={engine.num_layers}")

    # Get the main embed
    print(f"[smoke] loading embed (1.31 GB BF16)...")
    t0 = time.perf_counter()
    embed, head = load_embed_and_head(sf)
    print(f"[smoke] embed loaded in {time.perf_counter()-t0:.1f}s, shape {tuple(embed.shape)}")

    # Initial hidden states = embed(input_ids)
    hidden = embed[input_ids[0]]  # [S, hidden=2560]
    tokens_t = input_ids[0]
    print(f"[smoke] running 42-layer chain at policy {args.policy}...")
    t0 = time.perf_counter()
    final = run_chain(engine, hidden.unsqueeze(0), tokens_t, get_layer_types())
    elapsed = time.perf_counter() - t0
    print(f"[smoke] 42 layers done in {elapsed:.1f}s "
          f"({elapsed/42:.2f}s/layer)")

    # Final norm + lm_head
    norm_w = sf.get_tensor("model.language_model.norm.weight")
    var = final.float().pow(2).mean(-1, keepdim=True)
    final_normed = (final.float() * torch.rsqrt(var + 1e-6)) * norm_w.float()
    logits = final_normed @ head.float().T  # [1, S, vocab]
    last_logits = logits[0, -1, :]  # [vocab]
    top10 = torch.topk(last_logits, 10)
    top10_ids = top10.indices.tolist()
    top10_vals = top10.values.tolist()
    top10_strs = tok.convert_ids_to_tokens(top10_ids)
    print(f"\n[smoke] Top-10 next-token predictions for prompt {args.prompt!r}:")
    for tid, v, s in zip(top10_ids, top10_vals, top10_strs):
        try:
            print(f"  id={tid:6d}  logit={v:7.2f}  token={s!r}")
        except UnicodeEncodeError:
            # Windows console fallback
            print(f"  id={tid:6d}  logit={v:7.2f}  token=<unprintable>")

    out = {
        "policy": args.policy,
        "prompt": args.prompt,
        "input_token_ids": input_ids[0].tolist(),
        "elapsed_s": elapsed,
        "per_layer_s": elapsed / 42,
        "top10_token_ids": top10_ids,
        "top10_logits": top10_vals,
        "top10_token_strs": top10_strs,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[smoke] wrote {args.out}")


if __name__ == "__main__":
    main()
