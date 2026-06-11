"""
Read safetensors v1 metadata (JSON header at the start of the file).
v1 layout: [u64 header_len][JSON header][tensor data]
"""
import os, json
path = r'C:\Users\Zwmar\.lmstudio\hub\models--google--gemma-4-E4B\snapshots\a24c9379fd3839ae84e97f0b6aa3152fce9bd033\model.safetensors'
with open(path, 'rb') as f:
    hdr_len = int.from_bytes(f.read(8), 'little')
    hdr = f.read(hdr_len).decode('utf-8')
meta = json.loads(hdr)
print('format:', meta.get('__metadata__', {}).get('format'))
print('NUM_TENSORS:', sum(1 for k in meta if k != '__metadata__'))

# Per-key shapes
shapes = {}
for k, v in meta.items():
    if k == '__metadata__':
        continue
    shapes[k] = (v.get('shape'), v.get('dtype'))

from collections import Counter
prefixes = Counter()
for k in shapes:
    if '.layers.' in k:
        parts = k.split('.')
        prefixes['layer_with_' + parts[-2]] += 1
    else:
        prefixes[k] = 1

print('\n--- layer-internal tensor counts (across 42 layers) ---')
for p, c in sorted(prefixes.items()):
    print(f'  {c:4d}  {p}')

print('\n--- key shapes ---')
keys_to_show = [
    'model.language_model.layers.0.self_attn.q_proj.weight',
    'model.language_model.layers.0.self_attn.k_proj.weight',
    'model.language_model.layers.0.self_attn.v_proj.weight',
    'model.language_model.layers.0.self_attn.o_proj.weight',
    'model.language_model.layers.0.mlp.gate_proj.weight',
    'model.language_model.layers.0.mlp.up_proj.weight',
    'model.language_model.layers.0.mlp.down_proj.weight',
    'model.language_model.layers.0.per_layer_input_gate.weight',
    'model.language_model.layers.0.per_layer_projection.weight',
    'model.language_model.layers.0.post_per_layer_input_norm.weight',
    'model.language_model.layers.0.post_attention_layernorm.weight',
    'model.language_model.embed_tokens_per_layer.weight',
    'model.language_model.embed_tokens.weight',
    'model.language_model.lm_head.weight',
    'model.language_model.layers.5.self_attn.q_proj.weight',
    'model.language_model.layers.5.self_attn.k_proj.weight',
    'model.language_model.layers.5.self_attn.v_proj.weight',
    'model.language_model.layers.5.self_attn.o_proj.weight',
    'model.language_model.norm.weight',
    'model.language_model.layers.0.layer_scalar',
    'model.language_model.layers.5.layer_scalar',
]
for k in keys_to_show:
    print(f'  {shapes.get(k, "MISSING")}  {k}')

# Check shared KV
print('\n--- shared KV check ---')
k_keys = [k for k in shapes if k.endswith('.self_attn.k_proj.weight')]
v_keys = [k for k in shapes if k.endswith('.self_attn.v_proj.weight')]
print(f'  unique k_proj tensors: {len(k_keys)} (expect 42)')
print(f'  unique v_proj tensors: {len(v_keys)} (expect 42)')
import re
k_layers = sorted(int(re.search(r'layers\.(\d+)\.', k).group(1)) for k in k_keys)
print(f'  k layers: {k_layers}')

# Top-level non-language tensors
print('\n--- top-level language model tensors ---')
for k in shapes:
    if not k.startswith('model.language_model.layers.'):
        print(f'  {shapes[k]}  {k}')

# Total params
total = 0
for k, v in meta.items():
    if k == '__metadata__':
        continue
    n = 1
    for d in v.get('shape', []):
        n *= d
    total += n
print(f'\nTOTAL_PARAMS: {total/1e9:.3f} B')
