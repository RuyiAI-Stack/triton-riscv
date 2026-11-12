#!/usr/bin/env python3
"""Numeric simcheck driver for BOSCAME FPGA model 'electra-base-discriminator'.

Inputs: exported kernels and CPU oracles / reference tensors for each registered
case.

Outputs: pass/fail and max-error per operator without needing the FPGA board.

When used: after AOT export, before or instead of a board run, from the model
directory.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
from cases import describe, inventory  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from check_kernels import inputs, reference  # noqa: E402
from export import digest  # noqa: E402


def simulate(case, kernels, order="reverse"):
    artifacts = kernels / case["name"]
    module = artifacts / "kernel.boscame.mlir"
    manifest = json.loads((artifacts / "frontend.json").read_text())
    record = json.loads((artifacts / "build-result.json").read_text())
    if record.get("status") != "PASS" or record.get("stage") == "ttir":
        raise ValueError("a successful BOSCAME or later build is required")
    for key in (
        "name",
        "family",
        "symbol",
        "signature",
        "constexprs",
        "grid",
        "shape",
        "arguments",
        "argument_shapes",
    ):
        if manifest.get(key) != case.get(key):
            raise ValueError("stale metadata: " + key)
    if manifest["kernel_module_sha256"] != digest(case["kernel_source"]):
        raise ValueError("stale kernel source")
    if record.get("artifacts", {}).get(module.name) != digest(module):
        raise ValueError("unrecorded or stale emitted module")
    buddy = Path(os.environ.get("BUDDY_ROOT", str(REPO / "buddy-mlir")))
    sys.path.insert(0, str(buddy / "examples/FPGA-BOSCAME/tools"))
    from boscame_buffer_sim import Machine

    opt = os.environ.get("BUDDY_OPT", str(buddy / "build/bin/buddy-opt"))
    with tempfile.TemporaryDirectory(prefix="electra-boscame-sim-") as tmp:
        lowered = Path(tmp) / "kernel.loops.mlir"
        subprocess.run(
            [
                opt,
                str(module),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(lowered),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        machine = Machine(lowered, case["symbol"])
    arrays = inputs(case)
    wants = reference(case, arrays)
    initial = [arrays[arg["name"]] for arg in case["arguments"]]
    for name in wants:
        arrays[name].fill(-128 if arrays[name].dtype.kind == "i" else np.nan)
    outputs = [i for i, arg in enumerate(case["arguments"]) if arg["name"] in wants]
    guard = bytes([0xC7]) * 256
    buffers = {
        f"%arg{i}": bytearray(a.tobytes() + guard) for i, a in enumerate(initial)
    }
    original = {k: bytes(v) for k, v in buffers.items()}
    limits = {f"%arg{i}": a.nbytes for i, a in enumerate(initial)}
    for bank in machine.spm:
        bank[:] = bytes([0xD3]) * len(bank)
    points = list(itertools.product(*(range(x) for x in case["grid"])))
    if order == "reverse":
        points.reverse()
    count = len(initial)
    for point in points:
        scalars = {f"%arg{count + i}": x for i, x in enumerate([*case["grid"], *point])}
        machine.launch(buffers, scalars)
    actual = []
    for i, a in enumerate(initial):
        key = f"%arg{i}"
        if buffers[key][-len(guard) :] != guard:
            raise AssertionError("output guard modified: " + key)
        if i not in outputs and bytes(buffers[key]) != original[key]:
            raise AssertionError("input modified: " + key)
        actual.append(
            np.frombuffer(buffers[key], dtype=a.dtype, count=a.size).reshape(a.shape)
        )
    for name, start, end, action in machine.accesses:
        if name not in limits or not 0 <= start < end <= limits[name]:
            raise AssertionError(
                "access outside logical argument: " + str((name, start, end, action))
            )
        if action == "write" and int(name.removeprefix("%arg")) not in outputs:
            raise AssertionError("write to read-only input " + name)
        if case["family"] == "matmul_i8" and name == "%arg2" and action == "read":
            raise AssertionError("overwrite matmul read old C")
    maximum = 0.0
    for i in outputs:
        want = wants[case["arguments"][i]["name"]]
        got = actual[i]
        if got.dtype.kind in "iu":
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, atol=4e-4, rtol=4e-4)
        maximum = max(
            maximum,
            float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))),
        )
    return {
        "case": case["name"],
        "status": "STRICT_SIM_PASS",
        "programs": len(points),
        "order": order,
        "checked_elements": sum(w.size for w in wants.values()),
        "max_absolute_error": maximum,
        "ddr_accesses": len(machine.accesses),
        "dma_transfers": len(machine.dma_log),
        "guard_bytes_checked": len(guard) * len(buffers),
        "readonly_inputs_unchanged": True,
        "kernel_sha256": digest(case["kernel_source"]),
        "module_sha256": digest(module),
        "board_executed": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append", dest="cases")
    p.add_argument("--all", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "build/kernels")
    p.add_argument("--report", type=Path)
    p.add_argument("--order", choices=("forward", "reverse"), default="reverse")
    a = p.parse_args()
    cases = [describe(n) for n in a.cases] if a.cases else inventory()
    results = []
    for case in cases:
        try:
            r = simulate(case, a.output, a.order)
        except Exception as e:
            r = {
                "case": case["name"],
                "status": "FAIL",
                "reason": str(e),
                "board_executed": False,
            }
        print(r["status"], case["name"], r.get("reason", ""), flush=True)
        results.append(r)
    report = a.report or ROOT / "build/simcheck-report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    ok = all(r["status"] == "STRICT_SIM_PASS" for r in results)
    report.write_text(
        json.dumps(
            {
                "status": "PASS" if ok else "FAIL",
                "validation": "strict emitted BOSCAME DMA/SPM/REG interpreter; no FPGA execution",
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
