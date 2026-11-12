#!/usr/bin/env python3
"""Tokenize a real sentence pair and save official 12-layer XLNet FP32 outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    difference = got - want
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "cosine": float(
            np.vdot(got.ravel(), want.ravel())
            / (np.linalg.norm(got) * np.linalg.norm(want))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path, default=ROOT / "build/model-input")
    parser.add_argument("--lm-head", action="store_true")
    args = parser.parse_args()
    import torch
    from model_reference import forward
    from transformers import XLNetLMHeadModel, XLNetTokenizer

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    text = "The engineer verified the complete transformer model with real checkpoint weights."
    pair = "Every output tensor was returned for an independent numerical comparison."
    tokenizer = XLNetTokenizer(
        vocab_file=str(args.assets / "spiece.model"), do_lower_case=False
    )
    inputs = tokenizer(
        text,
        pair,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    assert tokenizer.padding_side == "left"
    model = XLNetLMHeadModel.from_pretrained(args.assets, local_files_only=True).eval()
    with torch.no_grad():
        official = model.transformer(
            **inputs, use_mems=False, output_hidden_states=True, return_dict=True
        )
        decomposition = forward(model, **inputs, lm_head=False)
    # Full-depth FP32 GEMM/einsum reassociation accumulates more rounding than
    # the one-layer unit fixture; retain the measured error in the manifest.
    torch.testing.assert_close(
        decomposition, official.last_hidden_state, atol=1e-4, rtol=1e-4
    )
    files = []

    def save(name, value, dtype):
        value = np.ascontiguousarray(value, dtype=dtype)
        path = args.output / (name + ".bin")
        path.write_bytes(value.tobytes())
        files.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "nbytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    for name in ("input_ids", "attention_mask", "token_type_ids"):
        save(name, inputs[name].numpy(), "<i4")
    for i, value in enumerate(official.hidden_states):
        save(
            "fp32_" + ("embedding" if i == 0 else f"layer_{i:02d}"),
            value.numpy(),
            "<f4",
        )
    save("fp32_last_hidden_state", official.last_hidden_state.numpy(), "<f4")
    if args.lm_head:
        with torch.no_grad():
            logits = model.lm_loss(official.last_hidden_state)
        save("fp32_logits", logits.numpy(), "<f4")
    report = {
        "model_id": "xlnet/xlnet-base-cased",
        "revision": "ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79",
        "text": text,
        "text_pair": pair,
        "padding_side": tokenizer.padding_side,
        "real_tokens": int(inputs["attention_mask"].sum()),
        "segment_ids": sorted(set(inputs["token_type_ids"].flatten().tolist())),
        "checkpoint_sha256": sha256(args.assets / "pytorch_model.bin"),
        "tokenizer_sha256": sha256(args.assets / "spiece.model"),
        "output": (
            "XLNetLMHeadModel.logits[1,128,32000]"
            if args.lm_head
            else "XLNetModel.last_hidden_state[1,128,768]"
        )
        + "; eval; bi; use_mems=False",
        "fp32_decomposition_vs_transformers": metrics(
            decomposition.numpy(), official.last_hidden_state.numpy()
        ),
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
