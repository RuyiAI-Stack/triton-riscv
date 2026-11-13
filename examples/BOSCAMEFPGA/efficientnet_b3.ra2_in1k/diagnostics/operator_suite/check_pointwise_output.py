#!/usr/bin/env python3
"""Check every downloaded pointwise result with an independent integer oracle."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = json.loads(args.result.read_text())
    if (
        result.get("state") != "completed"
        or result.get("execution_status") != "exited"
        or result.get("app_exit_code") != 0
    ):
        raise RuntimeError("actual board task did not complete successfully")
    if result.get("image_sha256") != manifest["bin_sha256"]:
        raise RuntimeError("task used a different frozen image")
    downloads = {item["name"]: Path(item["path"]) for item in result["downloads"]}
    if set(downloads) != {"Output", "Before", "After"}:
        raise RuntimeError("full output and both guard artifacts required")
    for item in result["downloads"]:
        data = Path(item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError("download hash differs: " + item["name"])
    spec = manifest["case_metadata"]["constexprs"]
    spatial, channels, outputs = (spec[key] for key in ("M", "K", "N"))
    x = (
        ((np.arange(spatial * channels, dtype=np.int64) * 13 + 7) % 251 - 125)
        .astype(np.int32)
        .reshape(spatial, channels)
    )
    weight = (
        ((np.arange(outputs * channels, dtype=np.int64) * 17 + 11) % 255 - 128)
        .astype(np.int32)
        .reshape(outputs, channels)
    )
    bias = np.arange(outputs, dtype=np.int32) % 31 - 15
    expected = (x @ weight.T + bias).astype("<i4")
    actual = np.frombuffer(downloads["Output"].read_bytes(), dtype="<i4").reshape(
        spatial, outputs
    )
    mismatch = np.flatnonzero(actual.ravel() != expected.ravel())
    guards = {
        name: bool(
            np.all(
                np.frombuffer(downloads[name].read_bytes(), dtype="<u4") == 0x513A7E29
            )
        )
        for name in ("Before", "After")
    }
    report = {
        "case": manifest["case"],
        "status": "PASS" if not len(mismatch) and all(guards.values()) else "FAIL",
        "task_id": result["task_id"],
        "fpga": result["fpga"],
        "image_sha256": manifest["bin_sha256"],
        "outputs": int(actual.size),
        "mismatches": int(len(mismatch)),
        "guards": guards,
        "max_abs": int(np.abs(actual.astype(np.int64) - expected).max()),
        "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "first_mismatch_offsets": mismatch[:16].tolist(),
        "bit_exact": not bool(len(mismatch)),
        "scope": "full nonzero public adapter GEMM plus C bias; independent INT8 integer oracle",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
