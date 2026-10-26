#!/usr/bin/env python3
"""Check the complete emitted LayerNorm against standard numerical references.

The candidate always executes all 197 programs in the strict BOSCAME model.
An optional baseline module executes the identical inputs and launch order;
its errors are evidence rather than an acceptance requirement. PyTorch adds a
float32 comparison when installed, while the float64 and sequential float32
NumPy references are always checked. This is a numerical correctness check,
not a performance measurement.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from boscame_paths import buddy_tools
from oracles import make_inputs, reference
from simcheck import SimulationBlocked, check_artifacts

from build import ROOT, argument_shapes, describe, digest, tool

sys.path.insert(0, str(buddy_tools()))
from boscame_buffer_sim import Machine  # noqa: E402

DEFAULT_CANDIDATE = ROOT / "build/kernels/layernorm_197x768/kernel.boscame.mlir"
RTOL = ATOL = 2e-5


def execute_module(case, arrays, module, label):
    """Run a real emitted module with persistent poisoned SPM and DDR guards."""
    module = Path(module)
    if not module.is_file():
        raise SimulationBlocked("missing emitted module: " + str(module))
    module_hash = digest(module)
    with tempfile.TemporaryDirectory(prefix="vit-layernorm-numerics-") as tmp:
        loops = Path(tmp) / "kernel.loops.mlir"
        process = subprocess.run(
            [
                tool("buddy-opt"),
                str(module),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(loops),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.returncode:
            raise SimulationBlocked(
                label + ": cannot prepare emitted module: " + process.stdout
            )
        machine = Machine(loops, case["symbol"])
    for bank in machine.spm:
        bank[:] = bytes([0xD3]) * len(bank)
    guard = bytes([0xC7]) * 1024
    buffers = {
        f"%arg{i}": bytearray(arrays[arg["name"]].tobytes() + guard)
        for i, arg in enumerate(case["arguments"])
    }
    original = {name: bytes(data) for name, data in buffers.items()}
    rows, width = case["shape"]
    gx, gy, gz = case["grid"]
    if (gx, gy, gz) != (rows, 1, 1):
        raise ValueError(
            "LayerNorm numerical check requires one program per row"
        )
    for launched, row in enumerate(reversed(range(rows)), 1):
        machine.launch(
            buffers,
            {
                "%arg4": gx,
                "%arg5": gy,
                "%arg6": gz,
                "%arg7": row,
                "%arg8": 0,
                "%arg9": 0,
            },
        )
        if launched == 1 or launched == rows or launched % 40 == 0:
            print(f"{label}: {launched}/{rows} programs", flush=True)
    for i, argument in enumerate(case["arguments"]):
        name = f"%arg{i}"
        if buffers[name][-len(guard) :] != guard:
            raise AssertionError(label + ": DDR guard modified: " + name)
        if argument["name"] != "Out" and bytes(buffers[name]) != original[name]:
            raise AssertionError(label + ": input modified: " + name)
    limits = {
        f"%arg{i}": arrays[arg["name"]].nbytes
        for i, arg in enumerate(case["arguments"])
    }
    for name, start, end, mode in machine.accesses:
        if name not in limits or not 0 <= start < end <= limits[name]:
            raise AssertionError(
                label
                + f": access outside logical DDR: {(name, start, end, mode)}"
            )
        if mode == "write" and name != "%arg3":
            raise AssertionError(label + ": write to read-only input: " + name)
    if digest(module) != module_hash:
        raise SimulationBlocked(
            label + ": emitted module changed during execution"
        )
    result = (
        np.frombuffer(buffers["%arg3"], dtype=np.float32, count=rows * width)
        .copy()
        .reshape(rows, width)
    )
    return result, {
        "programs": rows,
        "total_programs": math.prod(case["grid"]),
        "outputs": result.size,
        "emitted_module_sha256": module_hash,
        "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "order": "reverse",
        "ddr_accesses": len(machine.accesses),
        "dma_transfers": len(machine.dma_log),
    }


def numerical_error(actual, expected):
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if not np.isfinite(error).all():
        raise AssertionError("LayerNorm numerical comparison contains NaN/Inf")
    return {
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "outside_rtol_atol_2e5": int(
            np.count_nonzero(error > ATOL + RTOL * np.abs(expected))
        ),
        "elements": int(actual.size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path)
    parser.add_argument(
        "--candidate-module", type=Path, default=DEFAULT_CANDIDATE
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "build/layernorm-numerics.json"
    )
    args = parser.parse_args()
    case = describe("layernorm_197x768")
    arrays = make_inputs(case, argument_shapes(case))
    report = {
        "schema": 1,
        "status": "BLOCKED",
        "shape": case["shape"],
        "input_seed": 71,
        "input_distribution": "normal(0.2,1.1); affine Weight uniform(-1,2), Bias uniform(-0.5,0.5)",
        "epsilon": case["constexprs"]["EPS"],
        "acceptance": {"rtol": RTOL, "atol": ATOL},
        "candidate_kernel_sha256": digest(case["kernel_source"]),
        "input_sha256": {
            name: hashlib.sha256(value.tobytes()).hexdigest()
            for name, value in arrays.items()
            if name != "Out"
        },
        "results": {},
    }
    try:
        if args.candidate_module.resolve() == DEFAULT_CANDIDATE.resolve():
            check_artifacts(case, ROOT / "build/kernels")
        x = arrays["X"].astype(np.float64)
        centered = x - np.mean(x, axis=1, keepdims=True)
        variance = np.mean(centered * centered, axis=1, keepdims=True)
        normalized = centered / np.sqrt(variance + case["constexprs"]["EPS"])
        standard = normalized * arrays["Weight"].astype(np.float64) + arrays[
            "Bias"
        ].astype(np.float64)
        references = {
            "float64_standard": standard,
            "sequential_f32_oracle": reference(case, arrays)["Out"],
        }
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            report["torch"] = {
                "status": "UNAVAILABLE",
                "reason": "optional torch dependency is not installed",
            }
        else:
            with torch.no_grad():
                references["torch_float32"] = functional.layer_norm(
                    torch.from_numpy(arrays["X"]),
                    [case["shape"][-1]],
                    torch.from_numpy(arrays["Weight"]),
                    torch.from_numpy(arrays["Bias"]),
                    eps=case["constexprs"]["EPS"],
                ).numpy()
            report["torch"] = {
                "status": "AVAILABLE",
                "version": torch.__version__,
            }
            report["torch_version"] = torch.__version__
        values = {}
        modules = []
        if args.baseline_module is not None:
            modules.append(("before", args.baseline_module))
        modules.append(("after", args.candidate_module))
        for label, module in modules:
            values[label], metrics = execute_module(case, arrays, module, label)
            for name, expected in references.items():
                metrics[name] = numerical_error(values[label], expected)
            report["results"][label] = metrics
        after = report["results"]["after"]
        passed = (
            all(
                after[name]["outside_rtol_atol_2e5"] == 0 for name in references
            )
            and after["sequential_f32_oracle"]["max_abs"] == 0
        )
        report["status"] = "PASS" if passed else "FAIL"
        if not passed:
            report["reason"] = (
                "candidate exceeds standard reference tolerance or differs from the ordered f32 oracle"
            )
        if "before" in values:
            same_output = (
                report["results"]["before"]["output_sha256"]
                == after["output_sha256"]
            )
            report.update(
                classification=(
                    "bit-identical to supplied baseline"
                    if same_output
                    else "numerics change relative to supplied baseline; compare standard-reference errors"
                ),
                change="mask centered padded lanes to +0 before variance; reductions still divide by WIDTH=768",
                old_new_max_abs=float(
                    np.max(
                        np.abs(
                            values["before"].astype(np.float64)
                            - values["after"].astype(np.float64)
                        )
                    )
                ),
            )
        else:
            report["classification"] = (
                "correctness check against standard LayerNorm; reference reductions may round differently"
            )
    except (SimulationBlocked, OSError) as error:
        report["reason"] = str(error)
    except Exception as error:
        report.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
