#!/usr/bin/env python3
"""Independent NumPy W8A8 reference for all 12 XLNet content-stream layers.

Integer products are evaluated with binary64 BLAS. Every product and partial
sum is an integer below 2**53 (K<=3072, operands<=127), so the result is the
exact signed-i32 dot product, independently of AME. All restore/add/LN/softmax
steps explicitly round to f32. GELU uses scipy's mathematical erf, independent
of the target polynomial. This reference never reads FPGA results.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from prepare_model_input import metrics
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]


def quantize(x):
    x = np.asarray(x, dtype=np.float32)
    maximum = np.max(np.abs(x), axis=1)
    scales = np.where(maximum == 0, np.float32(1), maximum / np.float32(127)).astype(
        np.float32
    )
    scaled = x / scales[:, None]
    base = np.trunc(scaled).astype(np.int32)
    fraction = scaled - base.astype(np.float32)
    rounded = base + np.where(
        np.abs(fraction) >= np.float32(0.5), np.where(scaled >= 0, 1, -1), 0
    )
    return np.clip(rounded, -127, 127).astype(np.int8), scales


def product(x, q, scales):
    aq, a_scale = quantize(x)
    # These are exact integer sums, then the same ordered two f32 multiplies
    # as dequantize (int32 -> f32 -> row scale -> column scale).
    integer = (aq.astype(np.float64) @ q.astype(np.float64).T).astype(np.int32)
    return (integer.astype(np.float32) * a_scale[:, None]) * scales[None, :]


def dynamic(x, right):
    q, scale = quantize(right)
    return product(x, q, scale)


def layernorm(x, weight, bias):
    mean = np.cumsum(x, axis=1, dtype=np.float32)[:, -1] / np.float32(768)
    centered = x - mean[:, None]
    variance = np.cumsum(centered * centered, axis=1, dtype=np.float32)[
        :, -1
    ] / np.float32(768)
    inverse = np.float32(1) / np.sqrt(variance + np.float32(1e-12))
    return ((centered * inverse[:, None]) * weight) + bias


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "build/weights")
    p.add_argument("--model-input", type=Path, default=ROOT / "build/model-input")
    p.add_argument("--output", type=Path, default=ROOT / "build/quantized-reference")
    p.add_argument("--lm-head", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.weights / "manifest.json").read_text())
    tensors = {}
    for r in manifest["files"]:
        path = a.weights / (r["name"] + ".bin")
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("weight checksum differs: " + str(path))
        tensors[r["name"]] = np.fromfile(path, dtype=r["dtype"]).reshape(r["shape"])
    ids = np.fromfile(a.model_input / "input_ids.bin", dtype="<i4")
    mask = np.fromfile(a.model_input / "attention_mask.bin", dtype="<i4")
    segments = np.fromfile(a.model_input / "token_type_ids.bin", dtype="<i4")
    blocked = ((mask[None, :] == 0) & ~np.eye(128, dtype=bool)).astype(np.float32)
    different = segments[:, None] != segments[None, :]
    h = tensors["word_embedding"][ids].copy()
    relative = tensors["relative_positions"]
    records = []

    def save(i, hidden):
        name = (
            "embedding"
            if i == 0
            else "last_hidden_state"
            if i == 12
            else f"layer_{i:02d}"
        )
        path = a.output / (name + ".bin")
        path.write_bytes(np.ascontiguousarray(hidden, dtype="<f4").tobytes())
        reference_name = "fp32_embedding" if i == 0 else f"fp32_layer_{i:02d}"
        fp32 = np.fromfile(
            a.model_input / (reference_name + ".bin"), dtype="<f4"
        ).reshape(128, 768)
        record = {
            "name": name,
            "shape": [1, 128, 768],
            "dtype": "<f4",
            "nbytes": 393216,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "quantized_vs_fp32": metrics(hidden, fp32),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    save(0, h)
    for layer in range(12):
        b = f"layer{layer}."

        def w(name, base=b):
            return tensors[base + name]

        def dense(x, name):
            return product(x, w(name + ".q"), w(name + ".scale"))

        def heads(x):
            return x.reshape(x.shape[0], 12, 64).transpose(1, 0, 2).copy()

        q, k, v, r = (
            heads(dense(h, "q")),
            heads(dense(h, "k")),
            heads(dense(h, "v")),
            heads(dense(relative, "r")),
        )
        ac = np.stack([dynamic(q[j] + w("r_w_bias")[j], k[j]) for j in range(12)])
        raw_bd = np.stack([dynamic(q[j] + w("r_r_bias")[j], r[j]) for j in range(12)])
        bd = raw_bd.reshape(12, 256, 128)[:, 1:, :].reshape(12, 128, 255)[:, :, :128]
        ef2 = np.stack(
            [
                product(q[j] + w("r_s_bias")[j], w(f"seg{j}.q"), w(f"seg{j}.scale"))
                for j in range(12)
            ]
        )
        ef = np.take_along_axis(
            ef2, np.broadcast_to(different, (12, 128, 128)).astype(np.int64), axis=2
        )
        score = (((ac + bd) + ef) * np.float32(0.125)) - blocked[None] * np.float32(
            1e30
        )
        shifted = score - np.max(score, axis=2)[:, :, None]
        exponential = np.exp(shifted, dtype=np.float32)
        total = np.cumsum(exponential, axis=2, dtype=np.float32)[:, :, -1]
        probability = exponential / total[:, :, None]
        context = np.stack([dynamic(probability[j], v[j].T) for j in range(12)])
        merged = context.transpose(1, 0, 2).reshape(128, 768).copy()
        attn = layernorm(
            dense(merged, "o") + h, w("attn_norm.weight"), w("attn_norm.bias")
        )
        ff = dense(attn, "ffn1") + w("ffn1.bias")
        gelu = (np.float32(0.5) * ff) * (
            np.float32(1) + erf(ff * np.float32(0.7071067811865475))
        )
        ff2 = dense(gelu, "ffn2") + w("ffn2.bias")
        h = layernorm(ff2 + attn, w("ffn_norm.weight"), w("ffn_norm.bias"))
        save(layer + 1, h)
    if a.lm_head:
        logits = np.empty((128, 32000), dtype="<f4")
        for i, start in enumerate(range(0, 32000, 1024)):
            tile = (
                product(h, tensors[f"lm{i}.q"], tensors[f"lm{i}.scale"])
                + tensors[f"lm{i}.bias"]
            )
            valid = min(1024, 32000 - start)
            logits[:, start : start + valid] = tile[:, :valid]
        path = a.output / "logits.bin"
        path.write_bytes(logits.tobytes())
        fp32 = np.fromfile(a.model_input / "fp32_logits.bin", dtype="<f4").reshape(
            128, 32000
        )
        record = {
            "name": "logits",
            "shape": [1, 128, 32000],
            "dtype": "<f4",
            "nbytes": logits.nbytes,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "quantized_vs_fp32": metrics(logits, fp32),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "status": "REFERENCE_READY",
        "evidence": "independent local NumPy W8A8 reference; no device output read",
        "input_manifest_sha256": hashlib.sha256(
            (a.model_input / "manifest.json").read_bytes()
        ).hexdigest(),
        "weights_manifest_sha256": hashlib.sha256(
            (a.weights / "manifest.json").read_bytes()
        ).hexdigest(),
        "thresholds_declared_before_board_run": {
            "rmse_max": 0.025,
            "max_abs_max": 0.25,
            "cosine_min": 0.9995,
        },
        "files": records,
    }
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
