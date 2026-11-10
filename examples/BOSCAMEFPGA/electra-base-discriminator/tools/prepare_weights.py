#!/usr/bin/env python3
"""Pack an ELECTRA checkpoint into an aligned blob and a typed C binder.

FP32 tensors retain checkpoint [out,in] layouts; runtime primitives quantize
the selected matrices. This prepares artifacts without executing the model.
"""

import argparse
import hashlib
import json
from pathlib import Path
from transformers import ElectraForPreTraining

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--output", type=Path, default=ROOT / "build/weights")
    a = p.parse_args()
    model = ElectraForPreTraining.from_pretrained(
        a.assets, local_files_only=True
    ).eval()
    c = model.config
    if (
        c.hidden_size,
        c.embedding_size,
        c.intermediate_size,
        c.num_hidden_layers,
        c.num_attention_heads,
        c.vocab_size,
    ) != (768, 768, 3072, 12, 12, 30522):
        raise ValueError("checkpoint differs from fixed specialization")
    bindings = {
        "word_embeddings": "electra.embeddings.word_embeddings.weight",
        "position_embeddings": "electra.embeddings.position_embeddings.weight",
        "token_type_embeddings": "electra.embeddings.token_type_embeddings.weight",
        "embedding_norm.weight": "electra.embeddings.LayerNorm.weight",
        "embedding_norm.bias": "electra.embeddings.LayerNorm.bias",
    }
    layer_map = {
        "query": "attention.self.query",
        "key": "attention.self.key",
        "value": "attention.self.value",
        "attention_output": "attention.output.dense",
        "attention_norm": "attention.output.LayerNorm",
        "intermediate": "intermediate.dense",
        "output": "output.dense",
        "output_norm": "output.LayerNorm",
    }
    for i in range(12):
        for field, name in layer_map.items():
            for suffix in ("weight", "bias"):
                bindings[f"layers[{i}].{field}.{suffix}"] = (
                    f"electra.encoder.layer.{i}.{name}.{suffix}"
                )
    for field, name in [
        ("discriminator_dense", "dense"),
        ("discriminator_prediction", "dense_prediction"),
    ]:
        for suffix in ("weight", "bias"):
            bindings[f"{field}.{suffix}"] = f"discriminator_predictions.{name}.{suffix}"
    a.output.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    records = []
    with (a.output / "weights.bin").open("wb") as blob:
        for field, name in bindings.items():
            blob.write(b"\0" * ((-blob.tell()) % 64))
            data = (
                state[name]
                .detach()
                .cpu()
                .float()
                .contiguous()
                .numpy()
                .astype("<f4", copy=False)
                .tobytes()
            )
            records.append(
                {
                    "field": field,
                    "checkpoint_name": name,
                    "shape": list(state[name].shape),
                    "offset": blob.tell(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            blob.write(data)
        size = blob.tell()
    binder = [
        "/* Generated aligned FP32 checkpoint bindings. */",
        "#ifndef ELECTRA_WEIGHT_BINDINGS_H",
        "#define ELECTRA_WEIGHT_BINDINGS_H",
        '#include "model_runtime/model.h"',
        "static inline int electra_bind_weights(ElectraWeights *w, const void *blob, size_t bytes) {",
        f"  if (!w || !blob || ((uintptr_t)blob&63u) || bytes < {size}u) return -1;",
        "  const unsigned char *base = (const unsigned char *)blob;",
    ]
    binder += [
        f"  w->{r['field']} = (const float *)(base + {r['offset']}u);" for r in records
    ]
    binder += ["  return 0;", "}", "#endif"]
    (a.output / "weights.h").write_text("\n".join(binder) + "\n")
    with (a.output / "weights.bin").open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "model": "google/electra-base-discriminator",
        "dtype": "little-endian FP32",
        "alignment_bytes": 64,
        "bytes": size,
        "sha256": sha,
        "tensors": records,
        "board_executed": False,
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packed {len(records)} tensors, {size} bytes: {a.output}")


if __name__ == "__main__":
    main()
