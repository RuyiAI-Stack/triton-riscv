#!/usr/bin/env python3
"""Fetch tokenizer/config/weights from the requested mirror; never runs a board."""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "assets")
parser.add_argument("--revision", default="1ae76a97c7e84a4e640876a07453fccd636f0667")
args = parser.parse_args()
snapshot_download(
    repo_id="google/electra-base-discriminator",
    revision=args.revision,
    local_dir=args.output,
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ],
)
manifest = {
    "repo_id": "google/electra-base-discriminator",
    "endpoint": os.environ["HF_ENDPOINT"],
    "requested_revision": args.revision,
    "files": {},
}
for path in sorted(args.output.iterdir()):
    if path.is_file() and path.name != "asset-manifest.json":
        with path.open("rb") as stream:
            manifest["files"][path.name] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
(args.output / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
