#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import torch
from executorch.runtime import Runtime

def run_method(runtime, path: Path, *inputs):
    program = runtime.load_program(str(path))
    method = program.load_method("forward")
    result = method.execute([x.contiguous() for x in inputs])
    return result[0] if isinstance(result, (list, tuple)) else result

def main():
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: build_cached_cond_emb.py cond_speech_emb.pte cond_enc.pte "
            "cond_speech_tokens.bin speaker_emb.bin output.bin"
        )

    emb_pte = Path(sys.argv[1])
    enc_pte = Path(sys.argv[2])
    tokens_file = Path(sys.argv[3])
    speaker_file = Path(sys.argv[4])
    out_file = Path(sys.argv[5])

    ids = np.fromfile(tokens_file, dtype=np.int32)
    if ids.size != 150:
        raise RuntimeError("Expected 150 conditioning speech tokens, got " + str(ids.size))

    speaker = np.fromfile(speaker_file, dtype=np.float32)
    if speaker.size != 256:
        raise RuntimeError("Expected 256 speaker embedding values, got " + str(speaker.size))

    runtime = Runtime.get()
    cond_tokens = torch.from_numpy(ids).reshape(1, 150).to(dtype=torch.int32)
    speaker_t = torch.from_numpy(speaker).reshape(1, 256).float()
    emotion_t = torch.full((1, 1, 1), 0.5, dtype=torch.float32)

    cond_speech_emb = run_method(runtime, emb_pte, cond_tokens)
    if tuple(cond_speech_emb.shape) != (1, 150, 1024):
        raise RuntimeError(
            "Unexpected cond speech embedding shape: " + str(tuple(cond_speech_emb.shape))
        )

    cond_emb = run_method(runtime, enc_pte, speaker_t, cond_speech_emb, emotion_t)
    if tuple(cond_emb.shape) != (1, 34, 1024):
        raise RuntimeError(
            "Unexpected conditioning embedding shape: " + str(tuple(cond_emb.shape))
        )

    arr = cond_emb.detach().cpu().numpy().astype(np.float32, copy=False)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_file)
    print(
        "cached_cond_emb: shape="
        + str(arr.shape)
        + ", dtype="
        + str(arr.dtype)
        + ", bytes="
        + str(out_file.stat().st_size)
    )

if __name__ == "__main__":
    main()
