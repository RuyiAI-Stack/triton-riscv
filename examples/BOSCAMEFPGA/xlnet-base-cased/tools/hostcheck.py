#!/usr/bin/env python3
"""Execute actual Triton bodies in the CPU interpreter against independent NumPy.

This verifies source semantics only. It does not execute emitted RISC-V code.
"""

import os

os.environ["TRITON_INTERPRET"] = "1"
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from cases import describe, inventory  # noqa: E402 - local modules require the example path
from export import argument_shapes, digest  # noqa: E402 - local modules require the example path
from oracles import compare, make_inputs, reference  # noqa: E402 - local modules require the example path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append", dest="cases")
    p.add_argument("--report", type=Path, default=ROOT / "build/hostcheck-report.json")
    a = p.parse_args()
    torch.set_num_threads(1)
    records = []
    for case in [describe(n) for n in a.cases] if a.cases else inventory():
        record = {
            "case": case["name"],
            "source_sha256": digest(case["kernel_source"]),
            "status": "FAIL",
        }
        try:
            spec = importlib.util.spec_from_file_location(
                "xlnet_" + case["name"], case["kernel_source"]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            maximum = 0
            for stress in [False, True]:
                arrays = make_inputs(case, argument_shapes(case), stress=stress)
                # Source body and oracle own separate mutable output arrays.
                expected = reference(case, arrays)
                tensors = [
                    torch.from_numpy(arrays[arg["name"]].copy())
                    for arg in case["arguments"]
                ]
                getattr(module, case["kernel"])[tuple(case["grid"])](
                    *tensors, **case["constexprs"]
                )
                actual = {
                    arg["name"]: tensor.numpy()
                    for arg, tensor in zip(case["arguments"], tensors)
                }
                maximum = max(maximum, compare(case, actual, expected))
            record.update(status="PASS", max_abs_error=maximum, complete_grids=2)
        except Exception as error:  # noqa: BLE001 - Record failures and continue every case.
            record["reason"] = str(error)
        print(
            record["status"] + ": " + case["name"] + (" " + record.get("reason", "")),
            flush=True,
        )
        records.append(record)
    report = {
        "scope": "actual Triton CPU interpreter; independent NumPy references; no emitted/FPGA execution",
        "cases": records,
        "status": "PASS" if all(x["status"] == "PASS" for x in records) else "FAIL",
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
