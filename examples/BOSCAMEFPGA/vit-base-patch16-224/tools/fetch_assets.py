#!/usr/bin/env python3
"""Fetch the pinned ViT files from the Hugging Face mirror.

The script is deliberately explicit about the repository and files: graph
capture can be reproduced without a transformers cache or an implicit model
revision. Large files are streamed and resumed by urllib's HTTP server when
possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_ID = "google/vit-base-patch16-224"
FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
REVISION = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
EXPECTED_SHA256 = {
    "config.json": "abdf73cc4ce79495a83f32a8c9265daa7a1363f5a3efa4704c88a6e7f45ad19d",
    "preprocessor_config.json": "90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2",
    # SHA256 of the bytes served by the pinned mirror revision.  The mirror's
    # Xet ETag is different from the file digest, so keep the content hash
    # here rather than treating the ETag as SHA256.
    "model.safetensors": "1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d",
}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "buddy-mlir-for-fpga/vit-example"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=120)
    except Exception:
        # A stale partial file can make a server reject Range. Restart once.
        if start:
            partial.unlink(missing_ok=True)
            return download(url, destination)
        raise
    mode = "ab" if start and getattr(response, "status", 200) == 206 else "wb"
    with response, partial.open(mode) as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(partial, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--base-url", default="https://hf-mirror.com")
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        choices=FILES,
        help="download only selected files",
    )
    args = parser.parse_args()
    selected = tuple(args.files) if args.files else FILES
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in selected:
        path = args.output / name
        url = f"{args.base_url.rstrip('/')}/{MODEL_ID}/resolve/{args.revision}/{name}"
        expected = EXPECTED_SHA256[name] if args.revision == REVISION else None
        cached = path.is_file() and expected and digest_file(path) == expected
        if cached:
            print(f"verified cached {name}", flush=True)
        else:
            print(f"fetch {url} -> {path}", flush=True)
            download(url, path)
        digest = digest_file(path)
        if args.revision == REVISION and digest != EXPECTED_SHA256[name]:
            raise RuntimeError(
                f"sha256 mismatch for {name}: expected {EXPECTED_SHA256[name]}, got {digest}"
            )
        records.append(
            {
                "file": name,
                "url": url,
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": args.revision,
                "source": args.base_url,
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
