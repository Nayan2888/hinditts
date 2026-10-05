#!/usr/bin/env python3
"""
Extract conditioning tensors from conds.pt for Android.
Saves as raw binary files that can be loaded as Tensor on device.

Usage:
    python scripts/extract_conds.py [path/to/conds.pt]
"""

import sys
import struct
import numpy as np

def main():
    import torch

    conds_path = sys.argv[1] if len(sys.argv) > 1 else "conds.pt"
    output_dir = "app/src/main/assets"

    print(f"Loading {conds_path}...")
    data = torch.load(conds_path, map_location="cpu", weights_only=True)

    if isinstance(data, dict):
        for key, tensor in data.items():
            save_tensor(tensor, f"{output_dir}/{key}.bin", key)
    elif isinstance(data, (list, tuple)):
        names = ["speaker_emb", "cond_speech_tokens", "prompt_tokens"]
        for i, tensor in enumerate(data):
            name = names[i] if i < len(names) else f"tensor_{i}"
            save_tensor(tensor, f"{output_dir}/{name}.bin", name)
    else:
        save_tensor(data, f"{output_dir}/conds.bin", "conds")

    print("Done!")

def save_tensor(tensor, path, name):
    import torch
    arr = tensor.detach().cpu()
    print(f"  {name}: shape={list(arr.shape)}, dtype={arr.dtype}")

    if arr.dtype in (torch.float32, torch.float16):
        arr = arr.float().numpy()
        with open(path, "wb") as f:
            f.write(arr.tobytes())
    elif arr.dtype in (torch.int32, torch.int64, torch.long):
        arr = arr.int().numpy()
        with open(path, "wb") as f:
            f.write(arr.tobytes())
    else:
        arr = arr.float().numpy()
        with open(path, "wb") as f:
            f.write(arr.tobytes())

    print(f"  Saved to {path} ({arr.nbytes} bytes)")

if __name__ == "__main__":
    main()
