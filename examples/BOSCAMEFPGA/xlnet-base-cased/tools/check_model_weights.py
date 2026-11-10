#!/usr/bin/env python3
"""Audit every packed tensor against real checkpoint tensors using Torch."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--weights", type=Path, default=ROOT / "build/weights")
    p.add_argument("--output", type=Path, default=ROOT / "build/weights-audit.json")
    a = p.parse_args()
    import torch
    from transformers import XLNetLMHeadModel

    torch.set_num_threads(4)
    model = XLNetLMHeadModel.from_pretrained(a.assets, local_files_only=True).eval()
    manifest = json.loads((a.weights / "manifest.json").read_text())
    entries = {r["name"]: r for r in manifest["files"]}
    checked = []

    def equal(name, expected):
        r = entries[name]
        path = a.weights / (name + ".bin")
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == r["sha256"], name
        actual = np.frombuffer(raw, dtype=r["dtype"]).reshape(r["shape"])
        wanted = expected.detach().cpu().contiguous().numpy()
        np.testing.assert_array_equal(
            actual.view(np.uint8), wanted.view(np.uint8), err_msg=name
        )
        checked.append({"name": name, "sha256": r["sha256"], "nbytes": len(raw)})

    def matrix(name, value):
        value = value.detach().float().contiguous()
        maximum = value.abs().amax(1)
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
        ratio = value / scale[:, None]
        base = torch.trunc(ratio).to(torch.int32)
        residual = ratio - base.to(torch.float32)
        adjustment = torch.where(
            residual.abs() >= 0.5, torch.where(ratio >= 0, 1, -1), 0
        )
        q = torch.clamp(base + adjustment, -127, 127).to(torch.int8)
        equal(name + ".q", q)
        equal(name + ".scale", scale)

    equal("word_embedding", model.transformer.word_embedding.weight)
    for i, layer in enumerate(model.transformer.layer):
        b = f"layer{i}."
        attn = layer.rel_attn
        assert attn.layer_norm.eps == layer.ff.layer_norm.eps == 1e-12
        for name in ["q", "k", "v", "r"]:
            matrix(b + name, getattr(attn, name).reshape(768, 768).T)
        matrix(b + "o", attn.o.reshape(768, 768))
        for name in ["r_w_bias", "r_r_bias", "r_s_bias"]:
            equal(b + name, getattr(attn, name))
        for h in range(12):
            matrix(b + f"seg{h}", attn.seg_embed[:, h, :])
        for name, norm in [
            ("attn_norm", attn.layer_norm),
            ("ffn_norm", layer.ff.layer_norm),
        ]:
            equal(b + name + ".weight", norm.weight)
            equal(b + name + ".bias", norm.bias)
        for name, linear in [("ffn1", layer.ff.layer_1), ("ffn2", layer.ff.layer_2)]:
            matrix(b + name, linear.weight)
            equal(b + name + ".bias", linear.bias)
    if manifest["lm_head"]:
        assert (
            model.lm_loss.weight.data_ptr()
            == model.transformer.word_embedding.weight.data_ptr()
        )
        for i, start in enumerate(range(0, 32000, 1024)):
            valid = min(1024, 32000 - start)
            value = torch.zeros(1024, 768)
            bias = torch.zeros(1024)
            value[:valid] = model.lm_loss.weight[start : start + valid].detach()
            bias[:valid] = model.lm_loss.bias[start : start + valid].detach()
            matrix(f"lm{i}", value)
            equal(f"lm{i}.bias", bias)
    assert set(entries) == {r["name"] for r in checked} | {"relative_positions"}
    relative = np.fromfile(a.weights / "relative_positions.bin", dtype="<f4").reshape(
        256, 768
    )
    positions = torch.arange(128, -128, -1, dtype=torch.float32)
    inv = 1 / (10000 ** (torch.arange(0, 768, 2, dtype=torch.float32) / 768))
    x = positions[:, None] * inv[None, :]
    torch_relative = torch.cat([x.sin(), x.cos()], -1).numpy()
    report = {
        "status": "PASS",
        "checkpoint_sha256": hashlib.sha256(
            (a.assets / "pytorch_model.bin").read_bytes()
        ).hexdigest(),
        "packed_tensors_bit_exact_to_torch_audit": len(checked),
        "lm_head": manifest["lm_head"],
        "orientation": "q/k/v/r transpose [input,head,dim] to physical [output,input]; o physical unchanged",
        "relative_position_numpy_vs_torch_max_abs": float(
            np.abs(relative - torch_relative).max()
        ),
        "files": checked,
    }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
