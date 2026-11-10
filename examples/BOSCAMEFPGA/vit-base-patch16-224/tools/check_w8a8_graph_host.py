#!/usr/bin/env python3
"""
BOSCAME FPGA example 'check_w8a8_graph_host.py' under model
'vit-base-patch16-224'.

Inputs: neighboring kernel, launch, and Make files in this example tree.

Outputs: support for AOT compile, simcheck, or board execution.

When used: examples/BOSCAMEFPGA bring-up and numeric validation.
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

from model_contract import (
    require_attention_manifest,
    require_no_matrix_fallback,
    require_production_replacement,
)

ROOT = Path(__file__).resolve().parents[1]
from boscame_paths import llvm_lib_dir  # noqa: E402


def parse_arguments(source: str, entry: str):
    from buddy_mlir import ir

    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(source)
        function = next(
            (
                op
                for op in module.body.operations
                if op.operation.name == "func.func"
                and ir.StringAttr(op.attributes["sym_name"]).value == entry
            ),
            None,
        )
        if function is None or not function.regions[0].blocks:
            raise RuntimeError(f"entry @{entry} is missing from saved subgraph")
        result = []
        for value in function.regions[0].blocks[0].arguments:
            typ = value.type
            if not isinstance(typ, (ir.RankedTensorType, ir.MemRefType)) or any(
                dim <= 0 for dim in typ.shape
            ):
                raise RuntimeError(f"unsupported graph argument {typ}")
            dtype = str(typ.element_type)
            if dtype not in ("f32", "i8", "i32"):
                raise RuntimeError(f"unsupported graph element type {dtype}")
            result.append((tuple(typ.shape), dtype))
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
    adjustment = np.where(np.abs(fraction) >= 0.5, np.where(saturated >= 0, 1, -1), 0)
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
    parser.add_argument(
        "--oracle-dispatch",
        action="store_true",
        help="evaluate external kernel calls with tools/model_host_oracle.py",
    )
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
    require_production_replacement(report)
    manifest_identity = report["attention_primitive_manifest"]
    manifest_path = args.replacement / manifest_identity["file"]
    if (
        manifest_path.name != "attention-primitive-calls.json"
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != manifest_identity["sha256"]
    ):
        raise RuntimeError("attention primitive manifest identity is stale")
    manifest = json.loads(manifest_path.read_text())
    require_attention_manifest(report, manifest)
    if (
        manifest.get("adapter_sha256")
        != hashlib.sha256(
            (args.replacement / "vit_triton_adapters.c").read_bytes()
        ).hexdigest()
    ):
        raise RuntimeError(
            "generated descriptor adapters differ from the graph manifest"
        )
    if (
        report.get("workspace_buffer_policy_version") != 1
        or report.get("graph_mlir_compiled") is not True
    ):
        raise RuntimeError(
            "host execution requires current mutable-workspace graph plumbing"
        )
    identity = report.get("workspace_buffer_contract", {})
    contract_path = args.replacement / identity.get("file", "")
    if contract_path.name != "workspace-buffers.json" or hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest() != identity.get("sha256"):
        raise RuntimeError("mutable-workspace ABI contract is missing or stale")
    buffer_contract = json.loads(contract_path.read_text())
    if (
        buffer_contract.get("status") != "PASS"
        or buffer_contract.get("wrapper_abi", {}).get("status") != "PASS"
    ):
        raise RuntimeError("mutable-workspace graph ABI has not passed materialization")
    source = subgraph.read_text()
    require_no_matrix_fallback(source)
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
            raise RuntimeError(f"missing source parameter {linear['weight_param']}")
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
    # Exact argument roots come from GraphDriver's deduplicated signature.
    # Primitive read operands may have tensor-to-buffer/layout bridges; their
    # printed SSA spellings are not a positional ABI authority.
    records = {x["name"]: x for x in [*parameter_layout, *input_layout]}
    argument_roots = buffer_contract["argument_roots"]
    index_by_name = {name: index for index, name in enumerate(argument_roots)}
    if len(index_by_name) != len(signature) or not set(index_by_name).issubset(records):
        raise RuntimeError(
            "saved report and subgraph do not describe the same placeholder ABI"
        )
    value_by_name = {
        record["name"]: value for record, value in zip(input_layout, workspace_values)
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
        # Every descriptor consumed by a compiled kernel obeys the same
        # 64-byte base alignment as FPGA caller-owned buffers.
        backing = np.empty(value.nbytes + 63, dtype=np.uint8)
        offset = (-backing.ctypes.data) % 64
        aligned = (
            backing[offset : offset + value.nbytes]
            .view(value.dtype)
            .reshape(value.shape)
        )
        np.copyto(aligned, value)
        actual_inputs[index] = aligned

    graph.lower_to_top_level_ir()
    with ir.Location.unknown(graph._ctx):
        graph._imported_module = ir.Module.parse(source, context=graph._ctx)
    started = time.time()
    graph.lower_to_llvm_ir()
    lowering_seconds = round(time.time() - started, 3)
    llvm_lib = args.llvm_lib_dir
    libs = [
        llvm_lib / "libmlir_runner_utils.so",
        llvm_lib / "libmlir_c_runner_utils.so",
        args.host_library,
    ]
    for path in libs:
        require(path)
    dispatcher = None
    if args.oracle_dispatch:
        sys.path.insert(0, str(Path(__file__).parent))
        from model_host_oracle import HostOracleDispatcher

        dispatcher = HostOracleDispatcher(args.replacement, args.host_library)
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
    if dispatcher is not None and dispatcher.errors:
        raise RuntimeError("host kernel dispatch failed:\n" + dispatcher.errors[0])
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
        "execution": "host-only",
        "host_model_executed": True,
        "full_model_executed": False,
        "matrix_routes": report["matrix_routes"],
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
        "host_oracle": None if dispatcher is None else dispatcher.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
