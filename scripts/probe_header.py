"""Diagnose the model.safetensors corruption"""
import os
path = r'C:\Users\Zwmar\.lmstudio\hub\models--google--gemma-4-E4B\snapshots\a24c9379fd3839ae84e97f0b6aa3152fce9bd033\model.safetensors'
print('file size:', os.path.getsize(path))
with open(path, 'rb') as f:
    head = f.read(64)
    print('first 64 bytes hex:', head.hex())
    print('first 64 bytes ascii:', head[:32])
    f.seek(-64, 2)
    tail = f.read(64)
    print('last 64 bytes hex:', tail.hex())
    # Check if it has any safetensors JSON markers
    f.seek(0, 2)
    sz = f.tell()
    f.seek(sz - 200, 0)
    near_tail = f.read(200)
    print('last 200 bytes ascii:', near_tail)
    # Try detecting xet signature
    if head[:4] == b'XETB':
        print('FOUND XET HEADER')
    elif head[:3] == b'\x89\x50\x4e':
        print('FOUND PNG?? wrong file')
    else:
        print('not xet, not png. magic:', head[:8].hex())
