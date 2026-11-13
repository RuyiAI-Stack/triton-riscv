#!/usr/bin/env python3
"""Convert an XLNet checkpoint into explicit physical [N,K] W8A8 matrices.

Offline artifact preparation only; complete model linking/execution is separate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def quantize(value):
    maximum = np.max(np.abs(value), axis=1)
    scale = np.where(maximum == 0, np.float32(1), maximum / np.float32(127)).astype(
        "<f4"
    )
    ratio = value / scale[:, None]
    base = np.trunc(ratio).astype(np.int32)
    fraction = ratio - base.astype(np.float32)
    adjustment = np.where(
        np.abs(fraction) >= np.float32(0.5), np.where(ratio >= 0, 1, -1), 0
    )
    q = np.clip(base + adjustment, -127, 127).astype("i1")
    return q, scale


def relative_positions():
    pos = np.arange(128, -128, -1, dtype=np.float32)
    inv = np.float32(1) / (
        np.float32(10000) ** (np.arange(0, 768, 2, dtype=np.float32) / np.float32(768))
    )
    x = pos[:, None] * inv[None, :]
    return np.concatenate((np.sin(x), np.cos(x)), axis=1).astype("<f4")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--output", type=Path, default=ROOT / "build/weights")
    p.add_argument("--lm-head", action="store_true")
    a = p.parse_args()
    from transformers import XLNetLMHeadModel

    model = XLNetLMHeadModel.from_pretrained(a.assets, local_files_only=True).eval()
    actual = model.config
    for field, want in [
        ("d_model", 768),
        ("d_head", 64),
        ("n_head", 12),
        ("n_layer", 12),
        ("vocab_size", 32000),
        ("attn_type", "bi"),
        ("ff_activation", "gelu"),
    ]:
        if getattr(actual, field) != want:
            raise ValueError(f"unsupported {field}={getattr(actual, field)}")
    a.output.mkdir(parents=True, exist_ok=True)
    records = []

    def save(name, value):
        value = np.ascontiguousarray(value)
        target = a.output / (name + ".bin")
        target.write_bytes(value.tobytes())
        records.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )

    def array(t):
        return t.detach().cpu().float().numpy().astype("<f4")

    def matrix(name, t, transpose=False):
        w = array(t).reshape(768, 768)
        if transpose:
            w = w.T
        q, s = quantize(w)
        save(name + ".q", q)
        save(name + ".scale", s)

    save("word_embedding", array(model.transformer.word_embedding.weight))
    save("relative_positions", relative_positions())
    for i, layer in enumerate(model.transformer.layer):
        base = f"layer{i}"
        attn = layer.rel_attn
        # q/k/v/r[input,head,dim] -> [output,input]. o[output,head,dim] already physical [N,K].
        for name in ["q", "k", "v", "r"]:
            matrix(base + "." + name, getattr(attn, name), True)
        matrix(base + ".o", attn.o)
        for name in ["r_w_bias", "r_r_bias", "r_s_bias"]:
            save(base + "." + name, array(getattr(attn, name)))
        seg = array(attn.seg_embed).transpose(1, 0, 2)
        for h in range(12):
            q, s = quantize(seg[h])
            save(base + f".seg{h}.q", q)
            save(base + f".seg{h}.scale", s)
        for name, norm in [
            ("attn_norm", attn.layer_norm),
            ("ffn_norm", layer.ff.layer_norm),
        ]:
            save(base + "." + name + ".weight", array(norm.weight))
            save(base + "." + name + ".bias", array(norm.bias))
        for name, linear in [("ffn1", layer.ff.layer_1), ("ffn2", layer.ff.layer_2)]:
            q, s = quantize(array(linear.weight))
            save(base + "." + name + ".q", q)
            save(base + "." + name + ".scale", s)
            save(base + "." + name + ".bias", array(linear.bias))
    if a.lm_head:
        weight = array(model.lm_loss.weight)
        bias = array(model.lm_loss.bias)
        for tile, start in enumerate(range(0, 32000, 1024)):
            w = np.zeros((1024, 768), dtype="<f4")
            b = np.zeros(1024, dtype="<f4")
            valid = min(1024, 32000 - start)
            w[:valid] = weight[start : start + valid]
            b[:valid] = bias[start : start + valid]
            q, s = quantize(w)
            save(f"lm{tile}.q", q)
            save(f"lm{tile}.scale", s)
            save(f"lm{tile}.bias", b)
    (a.output / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "xlnet/xlnet-base-cased",
                "precision": "W8A8 matrices; FP32 original embedding",
                "lm_head": a.lm_head,
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(records)} tensors; no compiled model execution performed")


if __name__ == "__main__":
    main()
