#!/usr/bin/env python3
"""Fetch pinned XLNet assets; large weights are opt-in and download atomically."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "xlnet/xlnet-base-cased"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "assets")
    p.add_argument(
        "--revision",
        default=json.loads((ROOT / "model/manifest.json").read_text())["revision"],
    )
    p.add_argument("--weights", action="store_true")
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    meta = json.load(
        urllib.request.urlopen(
            f"{a.endpoint}/api/models/{MODEL}/revision/{a.revision or 'main'}",
            timeout=45,
        )
    )
    rev = meta["sha"]
    names = {x["rfilename"] for x in meta["siblings"]}
    wanted = [
        "config.json",
        "spiece.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    if a.weights:
        wanted += (
            ["model.safetensors"]
            if "model.safetensors" in names
            else ["pytorch_model.bin"]
        )
    manifest = {"model_id": MODEL, "revision": rev, "files": []}
    for name in wanted:
        if name not in names:
            continue
        target = a.output / name
        tmp = target.with_suffix(target.suffix + ".partial")
        with (
            urllib.request.urlopen(
                f"{a.endpoint}/{MODEL}/resolve/{rev}/{name}", timeout=120
            ) as source,
            tmp.open("wb") as dest,
        ):
            while data := source.read(1024 * 1024):
                dest.write(data)
        tmp.replace(target)
        with target.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["files"].append(
            {"name": name, "sha256": sha, "bytes": target.stat().st_size}
        )
        print(name, sha, flush=True)
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
