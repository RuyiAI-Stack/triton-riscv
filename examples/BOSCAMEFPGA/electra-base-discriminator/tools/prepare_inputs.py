#!/usr/bin/env python3
"""Prepare fixed-size INT32 input files for the C model entry on the host."""

import argparse
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--assets", type=Path, default=ROOT / "assets")
p.add_argument("--text", required=True)
p.add_argument("--pair")
p.add_argument("--output", type=Path, default=ROOT / "build/inputs")
a = p.parse_args()
tokenizer = AutoTokenizer.from_pretrained(a.assets, local_files_only=True)
encoded = tokenizer(
    a.text, a.pair, padding="max_length", truncation=True, max_length=128
)
a.output.mkdir(parents=True, exist_ok=True)
manifest = {}
for name, default in [
    ("input_ids", None),
    ("token_type_ids", [0] * 128),
    ("attention_mask", None),
]:
    value = np.asarray(encoded.get(name, default), dtype="<i4")
    if value.shape != (128,):
        raise ValueError("expected fixed [128] " + name)
    value.tofile(a.output / (name + ".bin"))
    manifest[name] = {
        "shape": [128],
        "dtype": "little-endian INT32",
        "file": name + ".bin",
    }
(a.output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(a.output)
