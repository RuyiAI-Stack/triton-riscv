#!/usr/bin/env python3
"""Validate the W8A8 Buddy graph on the host.

The replacement report is the ABI authority.  This checker obtains the
captured FP32 parameters in importer order, derives each added INT8 weight and
scale from the report's ``w8a8_linears`` records, allocates every caller-owned
workspace from ``input_layout``, and checks that this sequence exactly matches
the lowered ``@vit`` tensor signature before invoking the host engine.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from boscame_paths import llvm_lib_dir  # noqa: E402


def parse_arguments(source: str, entry: str):
    match = re.search(
        rf"func\.func\s+@{re.escape(entry)}\((.*?)\)\s*->", source, re.S
    )
    if match is None:
        raise RuntimeError(f"entry @{entry} is missing from saved subgraph")
    values = re.findall(r"%arg(\d+): tensor<([^>]+)>", match.group(1))
    if [int(i) for i, _ in values] != list(range(len(values))):
        raise RuntimeError("saved subgraph argument indices are not contiguous")
    result = []
    for _, typ in values:
        m = re.fullmatch(r"((?:\d+x)*)(f32|i8|i32)", typ)
        if m is None:
            raise RuntimeError(f"unsupported tensor type tensor<{typ}>")
        result.append(
            (tuple(int(x) for x in m.group(1).split("x") if x), m.group(2))
        )
    return result


def np_dtype(dtype: str):
    return {"f32": np.float32, "i8": np.int8, "i32": np.int32}[dtype]


def report_dtype(value: str) -> str:
    value = value.rsplit(".", 1)[-1]
    return {"Float32": "f32", "Int8": "i8", "Int32": "i32"}.get(value, value)


def quantize_rows(value: np.ndarray):
    value = np.ascontiguousarray(value, dtype=np.float32)
    rows = value.reshape(value.shape[0], -1)
    maximum = np.max(np.abs(rows), axis=1)
    scale = np.where(maximum == 0.0, 1.0, maximum / np.float32(127.0)).astype(
        np.float32
    )
    ratio = rows / scale[:, None]
    saturated = np.clip(ratio, np.float32(-127), np.float32(127))
    base = saturated.astype(np.int32)
    fraction = saturated - base.astype(np.float32)
    adjustment = np.where(
        np.abs(fraction) >= 0.5, np.where(saturated >= 0, 1, -1), 0
    )
    quantized = (base + adjustment).astype(np.int8).reshape(value.shape)
    return quantized, scale


def require(path: Path):
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--replacement", type=Path, default=ROOT / "build/replacement-w8a8"
    )
    parser.add_argument("--host-library", type=Path, required=True)
    parser.add_argument("--llvm-lib-dir", type=Path, default=llvm_lib_dir())
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/w8a8-graph-host-check.json"
    )
    parser.add_argument("--seed", type=int, default=0)
    # W8A8 is intentionally compared with a quantization tolerance.  The
    # checker still reports the raw max/mean error and top-1 match; these
    # defaults only decide whether the run is labelled PASS.  Callers that
    # need a stricter audit can override either value explicitly.
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--opt-level", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.llvm_lib_dir is None:
        parser.error("set --llvm-lib-dir, LLVM_LIB_DIR or LLVM_BIN")
    checkpoint = args.checkpoint or args.assets / "model.safetensors"
    subgraph = args.replacement / "subgraph0.triton.mlir"
    report_path = args.replacement / "triton-call-replacement.json"
    for path in (
        args.assets / "config.json",
        checkpoint,
        subgraph,
        report_path,
        args.host_library,
    ):
        require(path)
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS" or report.get("w8a8_linear_count") != 73:
        raise RuntimeError(
            "replacement report is not a complete 73-linear W8A8 graph"
        )
    source = subgraph.read_text()
    entry_match = re.search(r"func\.func\s+@([A-Za-z0-9_]+)\(", source)
    if entry_match is None:
        raise RuntimeError("cannot find subgraph entry")
    entry = entry_match.group(1)
    signature = parse_arguments(source, entry)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    import torch
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.ops import tosa
    from buddy_mlir import ir
    from buddy_mlir import runtime as rt
    from buddy_mlir.execution_engine import ExecutionEngine

    sys.path.insert(0, str(Path(__file__).parent))
    import import_model

    model, _, weights = import_model.load_model(
        args.assets / "config.json", checkpoint, torch.float32
    )
    rng = np.random.default_rng(args.seed)
    pixels_np = rng.uniform(-1.0, 1.0, (1, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        reference = model(torch.from_numpy(pixels_np)).detach().cpu().numpy()
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=import_model.capture_decompositions(),
        func_name="vit",
        enable_external_calls=True,
    )
    with torch.no_grad():
        graphs = compiler.importer(
            model, pixel_values=torch.zeros_like(torch.from_numpy(pixels_np))
        )
    if len(graphs) != 1:
        raise RuntimeError(f"expected one captured graph, got {len(graphs)}")
    graph = graphs[0]
    parameters = [
        np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
        for t in compiler.imported_params[graph]
    ]
    if len(parameters) != 200:
        raise RuntimeError(
            f"expected 200 original parameter tensors, got {len(parameters)}"
        )
    original = {f"arg{i + 1}_1": value for i, value in enumerate(parameters)}
    generated = {}
    for linear in report["w8a8_linears"]:
        value = original.get(linear["weight_param"])
        if value is None:
            raise RuntimeError(
                f"missing source parameter {linear['weight_param']}"
            )
        quantized, scale = quantize_rows(value)
        generated[linear["weight_param_name"]] = quantized
        generated[linear["scale_param_name"]] = scale

    parameter_layout = report["parameter_layout"]
    input_layout = report["input_layout"]
    image_records = [x for x in input_layout if x["name"] == "arg0_1"]
    if len(image_records) != 1:
        raise RuntimeError("input_layout must contain exactly one arg0_1 image")
    image_record = image_records[0]
    values = {**original, **generated}
    parameter_values = []
    for record in parameter_layout:
        value = values.get(record["name"])
        if value is None:
            raise RuntimeError(f"no value for parameter {record['name']}")
        expected_dtype = np_dtype(report_dtype(record["dtype"]))
        value = np.ascontiguousarray(value, dtype=expected_dtype)
        if list(value.shape) != record["shape"]:
            raise RuntimeError(f"parameter {record['name']} shape mismatch")
        parameter_values.append(value)
    workspace_values = []
    for record in input_layout:
        if record is image_record:
            workspace_values.append(pixels_np)
        else:
            workspace_values.append(
                np.zeros(
                    record["shape"],
                    dtype=np_dtype(report_dtype(record["dtype"])),
                )
            )
    # The lowered function interleaves parameter and workspace placeholders at
    # their first use.  Recover that exact order from the recorded external
    # call operands and the saved IR; never use placeholder order or equal
    # shapes as a positional guess.  An unused parameter is accepted only when
    # its shape/type identifies one unique remaining argument.
    records = {x["name"]: x for x in [*parameter_layout, *input_layout]}
    index_by_name = {}
    for call in report["replaced"]:
        match = re.search(
            r"call @" + re.escape(call["symbol"]) + r"\(([^)]*)\)", source
        )
        if match is None:
            raise RuntimeError(
                f"replacement call {call['symbol']} is missing from subgraph"
            )
        operands = [
            token.strip().split(":", 1)[0].strip()
            for token in match.group(1).split(",")
        ]
        for name, operand in zip(call.get("operands", []), operands):
            if name not in records or not operand.startswith("%arg"):
                continue
            index = int(operand[4:])
            previous = index_by_name.setdefault(name, index)
            if previous != index:
                raise RuntimeError(
                    f"placeholder {name} maps to both arg{previous} and arg{index}"
                )
    unassigned = set(range(len(signature))) - set(index_by_name.values())
    for name, record in records.items():
        if name in index_by_name:
            continue
        expected = (tuple(record["shape"]), report_dtype(record["dtype"]))
        candidates = [
            index for index in unassigned if signature[index] == expected
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"cannot uniquely map unused placeholder {name}: {expected}, candidates={candidates}"
            )
        index_by_name[name] = candidates[0]
        unassigned.remove(candidates[0])
    if len(index_by_name) != len(signature) or unassigned:
        raise RuntimeError(
            "saved report and subgraph do not describe the same placeholder ABI"
        )
    value_by_name = {
        record["name"]: value
        for record, value in zip(input_layout, workspace_values)
    }
    value_by_name.update(
        {
            record["name"]: value
            for record, value in zip(parameter_layout, parameter_values)
        }
    )
    value_by_name[image_record["name"]] = pixels_np
    actual_inputs = [None] * len(signature)
    for name, index in index_by_name.items():
        value = value_by_name[name]
        if tuple(value.shape) != signature[index][0] or value.dtype != np_dtype(
            signature[index][1]
        ):
            raise RuntimeError(f"ABI value mismatch for {name} at arg{index}")
        actual_inputs[index] = value

    graph.lower_to_top_level_ir()
    with ir.Location.unknown(graph._ctx):
        graph._imported_module = ir.Module.parse(source, context=graph._ctx)
    started = time.time()
    graph.lower_to_llvm_ir(enable_openmp=False)
    lowering_seconds = round(time.time() - started, 3)
    llvm_lib = args.llvm_lib_dir
    libs = [
        llvm_lib / "libmlir_runner_utils.so",
        llvm_lib / "libmlir_c_runner_utils.so",
        args.host_library,
    ]
    for path in libs:
        require(path)
    engine = ExecutionEngine(
        graph._imported_module,
        opt_level=args.opt_level,
        shared_libs=[str(path) for path in libs],
    )
    input_slots = []
    for value in actual_inputs:
        descriptor = rt.get_ranked_memref_descriptor(value)
        input_slots.append(ctypes.pointer(ctypes.pointer(descriptor)))
    output_struct = graph._output_descriptor()
    output_slot = ctypes.pointer(ctypes.pointer(output_struct))
    started = time.time()
    engine.invoke(entry, output_slot, *input_slots)
    invocation_seconds = round(time.time() - started, 3)
    result = np.array(
        rt.ranked_memref_to_numpy(ctypes.pointer(getattr(output_struct, "0"))),
        copy=True,
    )
    delta = np.abs(result - reference)
    tolerance = args.atol + args.rtol * np.abs(reference)
    ok = bool(np.all(np.isfinite(result)) and np.all(delta <= tolerance))
    summary = {
        "status": "PASS" if ok else "FAIL",
        "replacement_status": report["status"],
        "w8a8_linear_count": report["w8a8_linear_count"],
        "external_calls": report["graphs"]["forward"]["external_calls"],
        "entry": entry,
        "subgraph": str(subgraph),
        "subgraph_sha256": hashlib.sha256(subgraph.read_bytes()).hexdigest(),
        "parameter_count": len(parameter_values),
        "input_count": len(actual_inputs),
        "input_shape": list(pixels_np.shape),
        "output_shape": list(result.shape),
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "mismatched_elements": int(np.count_nonzero(delta > tolerance)),
        "reference_top1": int(reference.reshape(-1).argmax()),
        "compiled_top1": int(result.reshape(-1).argmax()),
        "top1_match": bool(
            reference.reshape(-1).argmax() == result.reshape(-1).argmax()
        ),
        "lowering_seconds": lowering_seconds,
        "invocation_seconds": invocation_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
