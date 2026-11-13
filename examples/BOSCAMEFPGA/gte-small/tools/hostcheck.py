#!/usr/bin/env python3
"""Run the real kernels in Triton's CPU interpreter against independent NumPy math.

This checks kernel numerical/layout semantics and the C ABI separately; it
never labels interpreter output as emitted BOSCAME or FPGA execution.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe, inventory  # noqa: E402 - local modules require the example path

sys.path.insert(0, str(ROOT / "tools"))
from export import adapter_source, argument_shapes  # noqa: E402 - local modules require the example path


def inputs(case, variant=0):
    arrays = []
    dtypes = {"f32": np.float32, "i32": np.int32, "i8": np.int8}
    for idx, (arg, shape) in enumerate(zip(case["arguments"], argument_shapes(case))):
        size = math.prod(shape)
        dtype = dtypes[arg["dtype"]]
        if idx in case["outputs"]:
            x = np.full(size, -77, dtype=dtype)
        else:
            i = np.arange(size, dtype=np.int64)
            if arg["dtype"] == "f32":
                x = ((i * 17) % 67 - 33).astype(dtype) * np.float32(0.015625)
            elif arg["dtype"] == "i8":
                x = ((i * 13) % 31 - 15).astype(dtype)
            else:
                x = ((i * 7) % 19 - 9).astype(dtype)
            if arg["name"] == "Ids":
                x = ((i * 227) % 30522).astype(dtype)
                x[-1] = 30521
            if arg["name"] == "PositionIds":
                x = i.astype(dtype)
            if arg["name"] == "TypeIds":
                x = ((i // 17) % 2).astype(dtype)
            if arg["name"] == "Mask":
                x = ((i < 113) & (i % 11 != 0)).astype(dtype)
                if variant:
                    x.fill(0)
            if arg["name"] == "Weight":
                x = (1.0 + (i % 7) * 0.01).astype(dtype)
            if arg["name"] in ("Row", "Column"):
                x = (0.0078125 + (i % 7) * 0.001953125).astype(dtype)
            if variant and case["family"] in (
                "per_token_quantization",
                "quantize_transpose",
                "l2_normalize",
            ):
                x.fill(0)
        arrays.append(x.reshape(shape))
    return arrays


def quantize(x):
    maximum = np.max(np.abs(x), axis=-1)
    scale = np.where(maximum == 0, 1, maximum / np.float32(127)).astype(np.float32)
    v = np.clip(x / scale[..., None], -127, 127)
    base = np.trunc(v).astype(np.int32)
    adjustment = np.where(
        np.abs(v - base.astype(np.float32)) >= 0.5, np.where(v >= 0, 1, -1), 0
    )
    return (base + adjustment).astype(np.int8), scale


def expected(case, a):
    family = case["family"]
    c = case["constexprs"]
    if family == "embedding_lookup":
        return [a[1][a[0]]]
    if family == "embedding_add":
        return [(a[0] + a[3][a[1]]) + a[4][a[2]]]
    if family == "matmul_i8":
        return [a[0].astype(np.int32) @ a[1].astype(np.int32).T]
    if family == "layernorm":
        x = a[0].astype(np.float64)
        y = (x - x.mean(-1, keepdims=True)) / np.sqrt(
            x.var(-1, keepdims=True) + c["EPS"]
        )
        return [(y * a[1] + a[2]).astype(np.float32)]
    if family in ("per_token_quantization", "quantize_transpose"):
        return list(quantize(a[0].T if family == "quantize_transpose" else a[0]))
    if family == "dequantization":
        return [
            ((a[0].astype(np.float32) * a[1][:, None]) * a[2][None, :])
            * np.float32(c["FACTOR"])
        ]
    if family == "bias_add":
        return [a[0] + a[1]]
    if family == "add":
        return [a[0] + a[1]]
    if family == "gelu":
        x = a[0].astype(np.float64)
        # PyTorch's independent CPU GELU uses the exact erf definition.
        return [
            torch.nn.functional.gelu(torch.from_numpy(x), approximate="none")
            .numpy()
            .astype(np.float32)
        ]
    if family == "head_layout":
        if c["SPLIT"]:
            return [a[0].reshape(128, 12, 32).transpose(1, 0, 2).copy()]
        return [a[0].transpose(1, 0, 2).reshape(128, 384).copy()]
    if family == "attention_mask":
        return [a[0] + np.where(a[1] != 0, np.float32(0), np.finfo(np.float32).min)]
    if family == "softmax":
        x = a[0].astype(np.float64)
        e = np.exp(x - x.max(-1, keepdims=True))
        return [(e / e.sum(-1, keepdims=True)).astype(np.float32)]
    if family == "mean_pool":
        denominator = max(float(a[1].sum()), 1e-9)
        return [
            (
                np.sum(a[0].astype(np.float64) * a[1][:, None], axis=0) / denominator
            ).astype(np.float32)
        ]
    if family == "l2_normalize":
        return [
            (a[0] / max(np.linalg.norm(a[0].astype(np.float64)), c["EPS"])).astype(
                np.float32
            )
        ]
    raise ValueError(family)


def kernelcheck(case):
    spec = importlib.util.spec_from_file_location(
        "gte_" + case["name"], case["kernel_source"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, case["kernel"])
    variants = (
        2
        if case["family"]
        in (
            "per_token_quantization",
            "quantize_transpose",
            "mean_pool",
            "l2_normalize",
            "attention_mask",
        )
        else 1
    )
    maximum = 0.0
    for variant in range(variants):
        arrays = inputs(case, variant)
        wants = expected(case, arrays)
        tensors = [torch.from_numpy(a.copy()) for a in arrays]
        fn[tuple(case["grid"])](*tensors, **case["constexprs"])
        for index, want in zip(case["outputs"], wants):
            got = tensors[index].numpy()
            if got.dtype.kind in "iu":
                np.testing.assert_array_equal(got, want)
            else:
                np.testing.assert_allclose(got, want, atol=4e-4, rtol=4e-4)
            maximum = max(
                maximum,
                float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))),
            )
        for i, arg in enumerate(arrays):
            if i not in case["outputs"]:
                np.testing.assert_array_equal(tensors[i].numpy(), arg)
    return {
        "case": case["name"],
        "status": "INTERPRETER_PASS",
        "variants": variants,
        "checked_elements_per_variant": case["checked_elements"],
        "max_absolute_error": maximum,
    }


def generate_adapters(cases, output):
    for case in cases:
        d = output / case["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter.c").write_text(adapter_source(case))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--abi-only", action="store_true")
    parser.add_argument("--kernel-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "build/hostcheck")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [describe(n) for n in args.cases] if args.cases else inventory()
    records = []
    if not args.abi_only:
        for case in cases:
            try:
                record = kernelcheck(case)
            except Exception as exc:  # noqa: BLE001 - Record each failure; the final report returns nonzero.
                record = {"case": case["name"], "status": "FAIL", "reason": str(exc)}
            records.append(record)
            print(record["status"], case["name"], record.get("reason", ""), flush=True)
    abi = 0
    if not args.kernel_only:
        adapters = args.output / "adapters"
        generate_adapters(cases, adapters)
        command = [
            sys.executable,
            str(ROOT / "tools/check_launch.py"),
            "--build-dir",
            str(adapters),
            "--output",
            str(args.output / "abi"),
        ]
        for case in cases:
            command += ["--case", case["name"]]
        abi = subprocess.run(command, check=False).returncode
    result = {
        "validation": "Triton CPU interpreter numerical checks and native C launch ABI; no emitted execution or FPGA run",
        "status": "PASS"
        if abi == 0 and all(r["status"] == "INTERPRETER_PASS" for r in records)
        else "FAIL",
        "cases": records,
        "abi_exit_code": abi,
        "board_executed": False,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
