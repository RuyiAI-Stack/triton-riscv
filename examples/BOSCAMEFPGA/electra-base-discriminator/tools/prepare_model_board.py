#!/usr/bin/env python3
"""Pack checkpoint W8A8 matrices and real tokenizer inputs for full-model FPGA execution."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "build/weights")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--output", type=Path, default=ROOT / "build/model-board")
    p.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    p.add_argument(
        "--pair", default="This sentence is used to validate the ELECTRA discriminator."
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((a.weights / "manifest.json").read_text())
    original = a.weights / "weights.bin"
    if sha(original) != source["sha256"]:
        raise ValueError("FP32 checkpoint blob hash differs from its manifest")
    records, bindings = [], []
    with original.open("rb") as src, (a.output / "params.bin").open("wb") as out:

        def append(field, data, dtype, shape, checkpoint):
            out.write(b"\0" * (-out.tell() % 64))
            payload = np.ascontiguousarray(data, dtype=dtype).tobytes()
            offset = out.tell()
            out.write(payload)
            records.append(
                dict(
                    field=field,
                    checkpoint_name=checkpoint,
                    offset=offset,
                    bytes=len(payload),
                    shape=shape,
                    dtype=dtype,
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
            ctype = "int8_t" if dtype == "|i1" else "float"
            bindings.append(f"  w->{field} = (const {ctype} *)(base + {offset}u);")

        for record in source["tensors"]:
            src.seek(record["offset"])
            payload = src.read(record["bytes"])
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError("tensor hash mismatch: " + record["field"])
            value = np.frombuffer(payload, dtype="<f4").reshape(record["shape"])
            field = record["field"]
            if value.ndim == 2 and (
                field.startswith("layers[") or field.startswith("discriminator_")
            ):
                maximum = np.max(np.abs(value), axis=-1)
                scale = np.where(
                    maximum == 0, np.float32(1), maximum / np.float32(127)
                ).astype("<f4")
                ratio = np.clip(value / scale[:, None], -127, 127)
                q = (
                    np.trunc(ratio)
                    + np.where(
                        np.abs(ratio - np.trunc(ratio)) >= 0.5, np.sign(ratio), 0
                    )
                ).astype("i1")
                prefix = field.removesuffix(".weight")
                append(
                    prefix + ".quantized",
                    q,
                    "|i1",
                    list(q.shape),
                    record["checkpoint_name"],
                )
                append(
                    prefix + ".scale",
                    scale,
                    "<f4",
                    list(scale.shape),
                    record["checkpoint_name"],
                )
                bindings.append(f"  w->{field} = 0;")
            else:
                append(field, value, "<f4", record["shape"], record["checkpoint_name"])
        total = out.tell()
    header = [
        '#include "model_runtime/model.h"',
        "static inline int electra_bind_board_weights(ElectraWeights *w, const void *blob, size_t bytes) {",
        f"  if (!w || !blob || ((uintptr_t)blob & 63u) || bytes != {total}u) return -1;",
        "  const unsigned char *base = blob;",
        *bindings,
        "  return 0;",
        "}",
    ]
    (a.output / "params.h").write_text("\n".join(header) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(a.assets, local_files_only=True)
    encoded = tokenizer(
        a.text, a.pair or None, padding="max_length", truncation=True, max_length=128
    )
    inputs = {}
    for name in ("input_ids", "token_type_ids", "attention_mask"):
        value = np.asarray(encoded.get(name, [0] * 128), dtype="<i4")
        if value.shape != (128,):
            raise ValueError("fixed specialization requires [128] inputs")
        path = a.output / (name + ".bin")
        value.tofile(path)
        inputs[name] = dict(
            shape=[128], dtype="<i4", sha256=sha(path), values=value.tolist()
        )
    manifest = dict(
        model="google/electra-base-discriminator",
        checkpoint=source["sha256"],
        asset_manifest=sha(a.assets / "asset-manifest.json"),
        params_bytes=total,
        params_sha256=sha(a.output / "params.bin"),
        quantization="row symmetric signed int8; ties away from zero; zero scale 1",
        text=a.text,
        pair=a.pair,
        tensors=records,
        inputs=inputs,
    )
    (a.output / "params-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"packed {total} bytes, {len(records)} tensors, {sum(encoded['attention_mask'])} valid tokens",
        flush=True,
    )


if __name__ == "__main__":
    main()
