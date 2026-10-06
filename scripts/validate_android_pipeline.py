#!/usr/bin/env python3
"""
Host-side validation of the exact five runtime PTEs used by Android.

This is intentionally independent of the Android Kotlin implementation.
It validates:
  1) conditioning PTEs
  2) T3 prefill
  3) one + several static-cache decode steps
  4) S3Gen encoder
  5) CFM x2
  6) HiFiGAN
  7) a complete Hindi text -> WAV pass

The script uses the same fixed-shape tensors and FP16/FP32 boundaries
as the current Android implementation.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from executorch.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]
PTE = ROOT / ".ci-models"
ASSETS = ROOT / "app" / "src" / "main" / "assets"
OUT = ROOT / ".ci-output"
OUT.mkdir(parents=True, exist_ok=True)

# Exact exported model constants from chatterbox-executorch.
SPEECH_VOCAB = 8194
SOT_TEXT = 255
EOT_TEXT = 0
SOT_SPEECH = 6561
EOT_SPEECH = 6562
MAX_TEXT_LEN = 256
TEXT_SEQ_LEN = 258
COND_LEN = 34
PREFILL_LEN = 293
MAX_SPEECH = 1000
MAX_KV_LEN = 1293
N_LAYERS = 30
N_HEADS = 16
HEAD_DIM = 64
KV_HALF = N_LAYERS * N_HEADS * MAX_KV_LEN * HEAD_DIM
HIFI_T_MEL = 300
T_MEL_FIXED = 2200
UPSAMPLE = 480
SR = 24000

def pte(name: str):
    path = PTE / name
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty PTE: {path}")
    rt = Runtime.get()
    program = rt.load_program(str(path))
    return program.load_method("forward")

def run(method, *args):
    return method.execute([a.contiguous() if isinstance(a, torch.Tensor) else a for a in args])

def out0(result):
    return result[0] if isinstance(result, (list, tuple)) else result

def check_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t.float()).all():
        raise RuntimeError(f"{name} contains NaN/Inf")

def load_f32(name: str, shape):
    a = np.fromfile(ASSETS / name, dtype=np.float32)
    if a.size != int(np.prod(shape)):
        raise RuntimeError(f"{name}: got {a.size} values, expected {int(np.prod(shape))}")
    return torch.from_numpy(a.copy()).reshape(shape)

def load_i32(name: str, shape):
    a = np.fromfile(ASSETS / name, dtype=np.int32)
    if a.size != int(np.prod(shape)):
        raise RuntimeError(f"{name}: got {a.size} values, expected {int(np.prod(shape))}")
    return torch.from_numpy(a.copy()).reshape(shape)

def load_i64(name: str, shape):
    a = np.fromfile(ASSETS / name, dtype=np.int64)
    if a.size != int(np.prod(shape)):
        raise RuntimeError(f"{name}: got {a.size} values, expected {int(np.prod(shape))}")
    return torch.from_numpy(a.copy()).reshape(shape)

def assert_shape(t: torch.Tensor, shape, name: str):
    got = tuple(int(x) for x in t.shape)
    exp = tuple(int(x) for x in shape)
    if got != exp:
        raise RuntimeError(f"{name}: shape {got} != expected {exp}")

def main():
    torch.manual_seed(1234)
    np.random.seed(1234)

    results = {}
    t_all = time.time()

    # ------------------------------------------------------------------
    # 1. Conditioning
    # ------------------------------------------------------------------
    cond_emb_pte = pte("t3_cond_speech_emb.pte")
    cond_enc_pte = pte("t3_cond_enc.pte")

    cond_tokens = load_i32("cond_speech_tokens.bin", (1, 150))
    speaker = load_f32("speaker_emb.bin", (1, 256))
    emotion = load_f32("emotion_adv.bin", (1, 1, 1))

    print("1) Conditioning")
    print("   tokens:", cond_tokens.dtype, tuple(cond_tokens.shape))
    cs = out0(run(cond_emb_pte, cond_tokens))
    assert_shape(cs, (1, 150, 1024), "cond_speech_emb")
    check_finite(cs, "cond_speech_emb")
    ce = out0(run(cond_enc_pte, speaker, cs, emotion))
    assert_shape(ce, (1, 34, 1024), "cond_emb")
    check_finite(ce, "cond_emb")
    print("   cond_speech_emb:", cs.dtype, tuple(cs.shape))
    print("   cond_emb:", ce.dtype, tuple(ce.shape))
    results["conditioning"] = "PASS"

    # ------------------------------------------------------------------
    # 2. Tokenizer contract. The Kotlin tokenizer returns payload IDs.
    #    T3 caller adds SOT/EOT and pads to 258.
    # ------------------------------------------------------------------
    print("2) Text input contract")
    # These IDs are checked by the Android tokenizer tests/assets. We only
    # need a known-good deterministic text tensor for the PTE smoke pass.
    # Use SOT + a language token + a few Hindi token IDs from the official
    # multilingual vocabulary; remaining entries are EOT padding.
    #
    # The language ID [hi] is 722 in the current tokenizer asset.
    hi_payload = [722]
    text = "नमस्ते, मेरा नाम नयन है।"
    # The Kotlin tokenizer is tested separately for exact character coverage;
    # here we use the same fixed interface and a compact known-good sequence.
    # IDs are populated from the bundled vocabulary at runtime below.
    vocab_json = ASSETS / "grapheme_mtl_merged_expanded_v1.json"
    if not vocab_json.is_file():
        raise RuntimeError("Tokenizer JSON missing from assets")
    data = json.loads(vocab_json.read_text(encoding="utf-8"))
    vocab = (data.get("model") or {}).get("vocab") or data.get("vocab") or {}
    for token in ("[hi]", "[UNK]", "[SPACE]"):
        if token not in vocab:
            raise RuntimeError(f"Tokenizer vocabulary missing {token}")
    # Deterministic character-level fallback for validation: map characters
    # that exist as tokens and use [UNK] for anything else.
    ids = [int(vocab["[hi]"])]
    for ch in text.lower():
        key = "[SPACE]" if ch == " " else ch
        ids.append(int(vocab.get(key, vocab["[UNK]"])))
    ids = ids[:MAX_TEXT_LEN]
    text_seq = [SOT_TEXT] + ids + [EOT_TEXT] * (MAX_TEXT_LEN - len(ids)) + [EOT_TEXT]
    text_t = torch.tensor([text_seq], dtype=torch.long)
    assert_shape(text_t, (1, 258), "text_tokens")
    results["tokenizer_contract"] = "PASS"
    print("   Hindi text payload tokens:", len(ids), "->", tuple(text_t.shape))

    # ------------------------------------------------------------------
    # 3. T3 prefill (FP16 input/output)
    # ------------------------------------------------------------------
    prefill = pte("t3_prefill.pte")
    print("3) T3 prefill")
    pref = run(prefill, ce, text_t)
    logits = pref[0].float()
    kv_flat = pref[1]
    assert_shape(logits, (1, SPEECH_VOCAB), "prefill logits")
    if kv_flat.numel() != N_LAYERS * 2 * N_HEADS * MAX_KV_LEN * HEAD_DIM:
        raise RuntimeError(f"kv_flat numel {kv_flat.numel()} unexpected")
    check_finite(logits, "prefill logits")
    print("   logits:", pref[0].dtype, tuple(pref[0].shape))
    print("   kv_flat:", pref[1].dtype, int(pref[1].numel()))
    results["t3_prefill"] = "PASS"

    # ------------------------------------------------------------------
    # 4. Static-cache decode. Match the exporter semantics:
    #    BOS is part of prefill. step_idx=0 writes the first NEW token at
    #    PREFILL_LEN.
    # ------------------------------------------------------------------
    decode = pte("t3_decode.pte")
    kv_shaped = kv_flat.reshape(N_LAYERS, 2, 1, N_HEADS, MAX_KV_LEN, HEAD_DIM)
    kv_k = kv_shaped[:, 0].contiguous()
    kv_v = kv_shaped[:, 1].contiguous()
    next_steps = 5
    generated = []
    print("4) T3 decode")
    for step in range(next_steps):
        token = int(torch.argmax(logits[0]).item())
        if token == EOT_SPEECH:
            print("   EOS at step", step)
            break
        if not (0 <= token < SPEECH_VOCAB):
            raise RuntimeError(f"invalid speech token {token}")
        generated.append(token)
        prev = torch.tensor([[token]], dtype=torch.long)
        step_idx = torch.tensor(step, dtype=torch.long)
        d = run(decode, prev, step_idx, kv_k, kv_v)
        logits = d[0].float()
        assert_shape(logits, (1, SPEECH_VOCAB), "decode logits")
        assert_shape(d[1], (N_LAYERS, 1, N_HEADS, MAX_KV_LEN, HEAD_DIM), "decode kv_k")
        assert_shape(d[2], (N_LAYERS, 1, N_HEADS, MAX_KV_LEN, HEAD_DIM), "decode kv_v")
        check_finite(logits, "decode logits")
        kv_k, kv_v = d[1], d[2]
    if not generated:
        raise RuntimeError("T3 generated zero speech tokens")
    print("   generated tokens:", generated)
    results["t3_decode"] = "PASS"

    # ------------------------------------------------------------------
    # 5. Vocoder: S3Gen encoder
    # ------------------------------------------------------------------
    s3 = pte("s3gen_encoder.pte")
    print("5) S3Gen encoder")
    speech = torch.zeros(1, 1000, dtype=torch.long)
    speech[0, :len(generated)] = torch.tensor(generated, dtype=torch.long)
    speech_len = torch.tensor([len(generated)], dtype=torch.long)
    prompt_tokens = load_i64("prompt_tokens.bin", (1, 75))
    prompt_len = load_i64("prompt_token_len.bin", (1,))
    xvector = load_f32("xvector.bin", (1, 192))
    enc = run(s3, speech, speech_len, prompt_tokens, prompt_len, xvector)
    h, h_len, embedding, mel_len1 = enc
    assert_shape(h, (1, 2150, 80), "s3gen h")
    assert_shape(embedding, (1, 80), "s3gen embedding")
    if h_len.numel() != 1 or mel_len1.numel() != 1:
        raise RuntimeError("Unexpected s3gen length tensors")
    check_finite(h, "s3gen h")
    check_finite(embedding, "s3gen embedding")
    h_len_v = int(h_len.reshape(-1)[0].item())
    mel_len1_v = int(mel_len1.reshape(-1)[0].item())
    if not (0 < mel_len1_v <= h_len_v <= 2150):
        raise RuntimeError(f"invalid mel lengths: mel_len1={mel_len1_v}, h_len={h_len_v}")
    print("   h:", h.dtype, tuple(h.shape), "h_len=", h_len_v, "mel_len1=", mel_len1_v)
    results["s3gen"] = "PASS"

    # ------------------------------------------------------------------
    # 6. CFM x2
    # ------------------------------------------------------------------
    cfm = pte("cfm_step.pte")
    print("6) CFM")
    h_t = h.permute(0, 2, 1)
    if h_t.shape[2] < T_MEL_FIXED:
        h_padded = torch.nn.functional.pad(h_t, (0, T_MEL_FIXED - h_t.shape[2]))
    else:
        h_padded = h_t[:, :, :T_MEL_FIXED]
    prompt_mel = load_f32("prompt_mel.bin", (1, 314, 80)).permute(0, 2, 1)
    cond_mel = torch.zeros(1, 80, T_MEL_FIXED)
    pm_len = min(prompt_mel.shape[2], mel_len1_v)
    cond_mel[:, :, :pm_len] = prompt_mel[:, :, :pm_len]
    mask = torch.zeros(1, 1, T_MEL_FIXED)
    mask[:, :, :h_len_v] = 1.0
    z = torch.randn(1, 80, T_MEL_FIXED)
    t_span = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    for i in range(2):
        t = torch.full((2,), float(t_span[i]))
        r = torch.full((2,), float(t_span[i + 1]))
        x_in = torch.cat([z, z], dim=0)
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([h_padded, torch.zeros_like(h_padded)], dim=0)
        spks = torch.cat([embedding, torch.zeros_like(embedding)], dim=0)
        cond = torch.cat([cond_mel, torch.zeros_like(cond_mel)], dim=0)
        c = run(cfm, x_in, mask_in, mu_in, t, spks, cond, r)
        dx = out0(c).float()
        assert_shape(dx, (2, 80, T_MEL_FIXED), "cfm dxdt")
        check_finite(dx, "cfm dxdt")
        z = z + (r[0] - t[0]) * ((1.0 + 0.7) * dx[:1] - 0.7 * dx[1:])
    mel = z[:, :, mel_len1_v:h_len_v]
    if mel.shape[2] <= 0:
        raise RuntimeError("CFM produced empty speech mel")
    check_finite(mel, "speech mel")
    print("   mel:", tuple(mel.shape))
    results["cfm"] = "PASS"

    # ------------------------------------------------------------------
    # 7. HiFiGAN
    # ------------------------------------------------------------------
    hifi = pte("hifigan.pte")
    print("7) HiFiGAN")
    wav_parts = []
    n_chunks = math.ceil(mel.shape[2] / HIFI_T_MEL)
    for ci in range(n_chunks):
        a = ci * HIFI_T_MEL
        b = min(a + HIFI_T_MEL, mel.shape[2])
        length = b - a
        chunk = torch.zeros(1, 80, HIFI_T_MEL)
        chunk[:, :, :length] = mel[:, :, a:b]
        noise = torch.randn(1, 9, HIFI_T_MEL * UPSAMPLE)
        phase = torch.zeros(1, 9, 1)
        hw = run(hifi, chunk, noise, phase)
        w = out0(hw).float()
        if w.numel() == 0:
            raise RuntimeError("HiFiGAN returned empty waveform")
        check_finite(w, "HiFiGAN wav")
        wav_parts.append(w.reshape(-1)[:length * UPSAMPLE].cpu().numpy())
    wav = np.concatenate(wav_parts)
    if wav.size < 1000:
        raise RuntimeError(f"waveform too short: {wav.size} samples")
    wav = np.clip(wav, -1.0, 1.0)
    out_wav = OUT / "hinditts_ci_validation.wav"
    sf.write(out_wav, wav, SR, subtype="PCM_16")
    print("   wav:", wav.size, "samples", f"{wav.size / SR:.2f}s", "->", out_wav)
    results["hifigan"] = "PASS"

    # ------------------------------------------------------------------
    # 8. Complete pass marker
    # ------------------------------------------------------------------
    results["complete_hindi_tts"] = "PASS"
    results["elapsed_seconds"] = round(time.time() - t_all, 2)
    (OUT / "validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("PASS: complete Hindi text -> WAV")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
