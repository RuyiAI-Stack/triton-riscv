#!/usr/bin/env python3
"""Encode one prefixed English text with the local fixed-shape host schedule."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model_reference import ROOT, SEQ, forward, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text", required=True, help="raw text without a query/passsage prefix"
    )
    parser.add_argument("--kind", choices=("query", "passage"), default="query")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--mode", choices=("fp32", "w8a8"), default="w8a8")
    parser.add_argument("--output", type=Path, default=ROOT / "build/encoded")
    args = parser.parse_args()
    text = args.text
    if text.startswith(("query: ", "passage: ")):
        raise ValueError(
            "supply unprefixed --text; --kind adds exactly one prefix before tokenization"
        )
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.assets, local_files_only=True)
    batch = tokenizer(
        args.kind + ": " + text,
        max_length=SEQ,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model, config = load_model(args.assets)
    ids = batch["input_ids"][0]
    mask = batch["attention_mask"][0].to(torch.int32)
    types = batch.get("token_type_ids", torch.zeros_like(batch["input_ids"]))[0]
    _, embedding, trace = forward(
        model.state_dict(), config, ids, mask, types, w8a8=args.mode == "w8a8"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    # Files contain little-endian contiguous values. Runtime callers must place
    # each operand at a separate 64-byte-aligned address before adapter calls.
    for name, value in [
        ("input_ids", ids.to(torch.int32)),
        ("attention_mask", mask),
        ("token_type_ids", types.to(torch.int32)),
        ("embedding", embedding),
    ]:
        (args.output / (name + ".bin")).write_bytes(value.numpy().tobytes())
    metadata = {
        "model": "intfloat/e5-small-v2",
        "backend": "host contract schedule",
        "mode": args.mode,
        "sequence_length": SEQ,
        "valid_tokens": mask.sum().item(),
        "prefix": args.kind + ": ",
        "embedding_shape": [384],
        "embedding_l2_norm": embedding.norm().item(),
        "operator_calls": len(trace),
        "board_executed": False,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
