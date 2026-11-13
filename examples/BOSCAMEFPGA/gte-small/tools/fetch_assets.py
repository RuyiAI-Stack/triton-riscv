#!/usr/bin/env python3
"""Fetch the pinned GTE checkpoint/tokenizer for optional real-weight host checks."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "model/manifest.json").read_text())
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ]
    if not args.metadata_only:
        files.append("model.safetensors")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in files:
        output = args.output / name
        temporary = output.with_suffix(output.suffix + ".download")
        url = f"{args.endpoint.rstrip('/')}/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
        # Always fetch the immutable revision; an existing main-revision cache
        # must not silently satisfy this pinned download request.
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        temporary.replace(output)
        hashes[name] = hashlib.sha256(output.read_bytes()).hexdigest()
        if name == "config.json" and json.loads(output.read_text()) != json.loads(
            (ROOT / "model/config.json").read_text()
        ):
            raise ValueError("pinned config does not match the operator specialization")
        print(name, hashes[name], flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps({**manifest, "sha256": hashes}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
