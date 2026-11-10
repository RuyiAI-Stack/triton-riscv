#!/usr/bin/env python3
"""Run the actual local Triton bodies on the CPU interpreter against NumPy.

This executes numerical kernel bodies, not BOSCAME code or FPGA instructions.
"""

import os

os.environ["TRITON_INTERPRET"] = "1"

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402
from export import argument_shapes  # noqa: E402


def inputs(case):
    rng = np.random.default_rng(42)
    arrays = {}
    for arg, shape in zip(case["arguments"], argument_shapes(case)):
        name, dtype = arg["name"], arg["dtype"]
        if dtype == "i8":
            value = rng.integers(-127, 128, shape, dtype=np.int8)
        elif dtype == "i32":
            value = rng.integers(-9000, 9000, shape, dtype=np.int32)
        else:
            value = rng.normal(0, 0.3, shape).astype(np.float32)
        arrays[name] = value
    family = case["family"]
    if family == "embedding":
        arrays["IDs"] = np.arange(128, dtype=np.int32) * 239 % 30522
        arrays["IDs"][-1] = 30521
        arrays["Types"] = np.arange(128, dtype=np.int32) % 2
    if family == "masked_softmax":
        arrays["Mask"] = ((np.arange(128) < 117) & (np.arange(128) % 11 != 3)).astype(
            np.int32
        )
    if family in ("per_token_quantization", "quantize_transpose"):
        rows = arrays["X"].T if family == "quantize_transpose" else arrays["X"]
        rows[0] = 0
        if len(rows) > 1:
            rows[1] = np.resize(
                np.array([-127, -2.5, -0.5, 0, 0.5, 2.5, 127], np.float32),
                rows.shape[1],
            )
    return arrays


def quantize(x):
    maximum = np.abs(x).max(axis=1)
    scale = np.where(maximum == 0, np.float32(1), maximum / np.float32(127))
    value = np.clip(x / scale[:, None], -127, 127)
    base = np.trunc(value).astype(np.int32)
    fraction = value - base.astype(np.float32)
    q = base + np.where(np.abs(fraction) >= 0.5, np.where(value >= 0, 1, -1), 0)
    return q.astype(np.int8), scale.astype(np.float32)


def reference(case, a):
    family = case["family"]
    if family == "matmul_i8":
        return {"C": a["A"].astype(np.int32) @ a["B"].astype(np.int32).T}
    if family in ("per_token_quantization", "quantize_transpose"):
        x = a["X"].T if family == "quantize_transpose" else a["X"]
        q, scale = quantize(x)
        return {"Q": q, "Scale": scale}
    if family == "dequantization":
        return {
            "Out": a["X"].astype(np.float32)
            * a["Row"][:, None]
            * a["Column"][None, :]
            * np.float32(case["output_factor"])
        }
    if family == "add":
        return {"Out": a["X"] + a["Y"]}
    if family == "bias_add":
        return {"Out": a["X"] + a["Bias"]}
    if family == "gelu":
        erf = np.frompyfunc(math.erf, 1, 1)
        return {
            "Out": (
                0.5
                * a["X"]
                * (1 + erf(a["X"].astype(np.float64) / math.sqrt(2)).astype(np.float64))
            ).astype(np.float32)
        }
    if family == "layernorm":
        x = a["X"].astype(np.float64)
        y = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-12)
        return {"Out": (y * a["Weight"] + a["Bias"]).astype(np.float32)}
    if family == "embedding":
        return {
            "Out": (a["Word"][a["IDs"]] + a["Segment"][a["Types"]])
            + a["Position"][:128]
        }
    if family == "pack_heads":
        return {"Out": a["X"].reshape(128, 12, 64).transpose(1, 0, 2)}
    if family == "unpack_heads":
        return {"Out": a["X"].transpose(1, 0, 2).reshape(128, 768)}
    if family == "masked_softmax":
        x = np.where(a["Mask"][None, None, :] != 0, a["X"], -np.inf)
        e = np.exp(x - x.max(-1, keepdims=True))
        return {"Out": e / e.sum(-1, keepdims=True)}
    raise ValueError("unhandled family " + family)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--report", type=Path, default=ROOT / "build/host-kernels.json")
    args = parser.parse_args()
    cases = inventory()
    if args.case:
        unknown = set(args.case) - {c["name"] for c in cases}
        if unknown:
            parser.error("unknown cases: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["name"] in args.case]
    results = []
    for case in cases:
        try:
            a = inputs(case)
            expected = reference(case, a)
            for name in expected:
                a[name].fill(-128 if a[name].dtype.kind == "i" else np.nan)
            tensors = {
                n: torch.from_numpy(np.ascontiguousarray(v)) for n, v in a.items()
            }
            spec = importlib.util.spec_from_file_location(
                "electra_" + case["name"], case["kernel_source"]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            getattr(module, case["kernel"])[tuple(case["grid"])](
                *(tensors[arg["name"]] for arg in case["arguments"]),
                **case["constexprs"],
            )
            checked = 0
            for name, want in expected.items():
                got = tensors[name].numpy()
                if got.dtype.kind == "i":
                    np.testing.assert_array_equal(got, want)
                else:
                    np.testing.assert_allclose(got, want, atol=3e-5, rtol=3e-5)
                checked += got.size
            record = {"case": case["name"], "status": "PASS", "checked": checked}
        except Exception as error:
            record = {"case": case["name"], "status": "FAIL", "reason": str(error)}
        results.append(record)
        print(record["status"], case["name"], record.get("reason", ""), flush=True)
    report = {
        "validation": "actual Triton CPU interpreter vs independent NumPy",
        "board_executed": False,
        "cases": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return int(any(r["status"] != "PASS" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
