#!/usr/bin/env python3
"""Independently audit packed board matrices/scales against the FP32 checkpoint blob."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model_reference import quantize
from prepare_model_board import ROOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=ROOT / "build/model-board")
    p.add_argument("--weights", type=Path, default=ROOT / "build/weights")
    a = p.parse_args()
    board = json.loads((a.build / "params-manifest.json").read_text())
    original = json.loads((a.weights / "manifest.json").read_text())
    assert sha(a.weights / "weights.bin") == original["sha256"] == board["checkpoint"]
    assert sha(a.build / "params.bin") == board["params_sha256"]
    source = {r["field"]: r for r in original["tensors"]}
    records, cache = [], {}
    torch.set_num_threads(2)
    with (
        (a.weights / "weights.bin").open("rb") as old,
        (a.build / "params.bin").open("rb") as new,
    ):
        for item in board["tensors"]:
            field = item["field"]
            prefix = field.rsplit(".", 1)[0]
            if field.endswith((".quantized", ".scale")):
                if prefix not in cache:
                    record = source[prefix + ".weight"]
                    old.seek(record["offset"])
                    x = (
                        np.frombuffer(old.read(record["bytes"]), dtype="<f4")
                        .copy()
                        .reshape(record["shape"])
                    )
                    q, scale = quantize(torch.from_numpy(x))
                    cache[prefix] = {
                        "quantized": q.to(torch.int8).numpy().tobytes(),
                        "scale": scale.numpy().astype("<f4").tobytes(),
                    }
                expected = cache[prefix][field.rsplit(".", 1)[-1]]
            else:
                record = source[field]
                old.seek(record["offset"])
                expected = old.read(record["bytes"])
            new.seek(item["offset"])
            actual = new.read(item["bytes"])
            equal = actual == expected
            if hashlib.sha256(actual).hexdigest() != item["sha256"]:
                raise ValueError("packed tensor manifest mismatch: " + field)
            records.append(
                dict(
                    field=field,
                    bytes=len(actual),
                    bit_exact=equal,
                    sha256=item["sha256"],
                )
            )
    report = dict(
        status="PASS" if all(x["bit_exact"] for x in records) else "FAIL",
        matrices=len(cache),
        tensors=len(records),
        params_sha256=board["params_sha256"],
        source_checkpoint_blob_sha256=original["sha256"],
        records=records,
        scope="independent Torch row quantization vs NumPy-packed board bytes; retained FP32 tensors compare directly",
    )
    (a.build / "weights-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
