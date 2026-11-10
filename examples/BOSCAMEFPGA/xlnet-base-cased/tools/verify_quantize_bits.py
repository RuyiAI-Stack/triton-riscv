#!/usr/bin/env python3
"""Bitwise local checks of scheduler-returned standard XLNet quantizer trials.

The inputs below are the fixed values in each unmodified launch.c.  This adds
strict Scale checks to the original host replay oracle, which permits a small
floating tolerance for per-token Scale.  No board-side oracle is executed.
Raw max is not exported by these production quantizers; it is tested separately
by the compiler max probe, rather than inferred from Scale here.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(case, trial):
    if case == "quantize_transpose_128x64":
        i = np.arange(128)
        base = np.where(
            i == 0, 127, np.where(i == 1, -127, (i % 29 - 14) * 0.25)
        ).astype("<f4")
        j = np.arange(64)
        factor = np.where(j % 2, np.float32(1.137), np.float32(1)).astype("<f4")
        value = (base[:, None] * factor[None, :]).astype("<f4")
        value[:, j % 5 == 0] = np.float32(0)
        return value.T.copy()
    match = re.fullmatch(r"quantize_(\d+)x(\d+)", case)
    if not match:
        raise ValueError("unsupported standard quantizer: " + case)
    m, k = map(int, match.groups())
    j = np.arange(k)
    base = np.where(
        j == 0,
        127,
        np.where(j == 1, -127, ((j * 13) % 31 - 15) * 0.25 + (j % 3) * 0.0625),
    ).astype("<f4")
    i = np.arange(m)
    factor = np.where(i % 2, np.float32(1.137), np.float32(1)).astype("<f4")
    value = (factor[:, None] * base[None, :]).astype("<f4")
    if trial == 0:
        value[i % 5 == 0] = np.float32(0)
    return value


def reference(value):
    maximum = np.max(np.abs(value), axis=1)
    scales = np.where(maximum == 0, np.float32(1), maximum / np.float32(127)).astype(
        "<f4"
    )
    ratio = np.clip(value / scales[:, None], np.float32(-127), np.float32(127))
    base = np.trunc(ratio).astype(np.int32)
    fraction = ratio - base.astype("<f4")
    q = (
        base
        + np.where(np.abs(fraction) >= np.float32(0.5), np.where(ratio >= 0, 1, -1), 0)
    ).astype("i1")
    return q, scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = args.case or sorted(
        p.name for p in args.runs.glob("quantize_*") if p.is_dir()
    )
    if not cases:
        parser.error("no quantizer result directories")
    rows = []
    for case in cases:
        directory = args.runs / case
        record = json.loads((directory / "result.json").read_text())
        manifest = json.loads((directory / "artifacts/manifest.json").read_text())
        assert manifest["id"] == record["task_id"]
        assert record["state"] == "completed" and record["execution_status"] == "exited"
        assert record["app_exit_code"] == record["exit_code"] == 0
        launch = ROOT / case / "launch.c"
        source = launch.read_text()
        if case != "quantize_transpose_128x64":
            assert "(j * 13) % 31 - 15" in source and "(j % 3) * 0.0625f" in source
            assert "(trial == 0 && (i % 5) == 0)" in source
        assert "1.137f : 1.0f" in source
        trials = 1 if case == "quantize_transpose_128x64" else 2
        expected = {}
        for trial in range(trials):
            q, scale = reference(inputs(case, trial))
            expected[f"r{trial}_Q"] = q
            expected[f"r{trial}_Scale"] = scale
        assert {x["name"] for x in manifest["artifacts"]} == set(expected)
        checks = []
        for item in manifest["artifacts"]:
            path = directory / "artifacts" / (item["name"] + ".bin")
            assert sha(path) == item["sha256"] and path.stat().st_size == item["nbytes"]
            actual = np.fromfile(path, dtype=item["dtype"]).reshape(item["shape"])
            want = expected[item["name"]]
            assert actual.shape == want.shape
            bits = np.uint32 if actual.dtype.kind == "f" else np.uint8
            bad = np.argwhere(actual.view(bits) != want.view(bits))
            checks.append(
                {
                    "name": item["name"],
                    "bit_exact": len(bad) == 0,
                    "different_elements": len(bad),
                    "actual_sha256": sha(path),
                    "expected_sha256": hashlib.sha256(want.tobytes()).hexdigest(),
                    "first_differences": [
                        {
                            "index": index.tolist(),
                            "actual_bits": hex(int(actual.view(bits)[tuple(index)])),
                            "expected_bits": hex(int(want.view(bits)[tuple(index)])),
                        }
                        for index in bad[:8]
                    ],
                }
            )
        rows.append(
            {
                "case": case,
                "task_id": record["task_id"],
                "fpga": record["fpga"],
                "launch_sha256": sha(launch),
                "checks": checks,
                "status": "PASS" if all(c["bit_exact"] for c in checks) else "FAIL",
            }
        )
    passed = all(r["status"] == "PASS" for r in rows)
    report = {
        "status": "PASS" if passed else "FAIL",
        "comparison": "bitwise Q and FP32 Scale",
        "raw_max_captured": False,
        "board_oracle_executed": False,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(rows)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
