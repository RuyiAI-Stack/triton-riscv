#!/usr/bin/env python3
"""Run a Buddy ViT subgraph with the Triton host bridge and compare logits.

The replacement stage writes the exact Buddy subgraph used by the model entry.
This check JIT lowers that saved graph, passes the captured checkpoint tensors
in Buddy's argument order, and compares its logits with the same PyTorch
wrapper and FP32 pixel tensor.  It is a host numerical check, not an FPGA run.
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


def require_paths(paths):
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")


def subgraph_argument_shapes(source, entry):
    """Read the saved tensor ABI before passing descriptors to the JIT."""
    match = re.search(rf"func\.func\s+@{re.escape(entry)}\(", source)
    if match is None:
        raise RuntimeError(f"entry @{entry} is missing from the saved subgraph")
    end = source.find(") ->", match.end())
    if end < 0:
        raise RuntimeError(f"cannot parse the argument list of @{entry}")
    signature = source[match.end() : end]
    arguments = re.findall(r"%arg(\d+): tensor<([^>]+)>", signature)
    if not arguments or [int(index) for index, _ in arguments] != list(
        range(len(arguments))
    ):
        raise RuntimeError(
            f"@{entry} has an unsupported or incomplete tensor argument list"
        )
    shapes = []
    for index, tensor_type in arguments:
        dimensions = re.fullmatch(r"((?:\d+x)*)f32", tensor_type)
        if dimensions is None:
            raise RuntimeError(
                f"@{entry} argument {index} has unsupported type tensor<{tensor_type}>"
            )
        shapes.append(
            tuple(int(dim) for dim in dimensions.group(1).split("x") if dim)
        )
    return shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--replacement", type=Path, default=ROOT / "build/replacement-fp32"
    )
    parser.add_argument(
        "--subgraph",
        type=Path,
        help="override saved subgraph (use import/subgraph0.mlir for a Buddy baseline)",
    )
    parser.add_argument(
        "--import-dir",
        type=Path,
        help="verify captured parameter order against import/params.json",
    )
    parser.add_argument("--host-library", type=Path, required=True)
    parser.add_argument("--llvm-lib-dir", type=Path, default=llvm_lib_dir())
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="diagnose a PARTIAL replacement before coverage is complete",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/host-graph-check.json"
    )
    parser.add_argument("--opt-level", type=int, choices=range(4), default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.llvm_lib_dir is None:
        parser.error("set --llvm-lib-dir, LLVM_LIB_DIR or LLVM_BIN")
    args.checkpoint = args.checkpoint or args.assets / "model.safetensors"
    subgraph = args.subgraph or args.replacement / "subgraph0.triton.mlir"
    subgraph_source = subgraph.read_text()
    match = re.search(r"func\.func\s+@([A-Za-z0-9_]+)\(", subgraph_source)
    if match is None:
        raise SystemExit(f"cannot find entry function in {subgraph}")
    entry = match.group(1)
    replacement_report = args.replacement / "triton-call-replacement.json"
    require_paths(
        [
            args.assets / "config.json",
            args.checkpoint,
            subgraph,
            replacement_report,
            args.host_library,
        ]
    )
    report = json.loads(replacement_report.read_text())
    if (
        args.subgraph is None
        and report.get("status") != "PASS"
        and not args.allow_partial
    ):
        raise SystemExit(
            f"replacement status is {report.get('status')}, expected PASS"
        )

    # libomp and MLIR's runner libraries must be present before the JIT engine
    # starts.  The supplied Triton bridge exports all graph external calls.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
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
        args.assets / "config.json", args.checkpoint, torch.float32
    )
    rng = np.random.default_rng(args.seed)
    pixels_np = rng.uniform(-1.0, 1.0, (1, 3, 224, 224)).astype(np.float32)
    pixels = torch.from_numpy(pixels_np)
    with torch.no_grad():
        reference = model(pixels).detach().cpu().numpy()
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=import_model.capture_decompositions(),
        func_name="vit",
        enable_external_calls=True,
    )
    with torch.no_grad():
        graphs = compiler.importer(model, pixel_values=torch.zeros_like(pixels))
    if len(graphs) != 1:
        raise RuntimeError(f"expected one captured graph, got {len(graphs)}")
    graph = graphs[0]
    parameters = [
        np.ascontiguousarray(t.detach().cpu().numpy())
        for t in compiler.imported_params[graph]
    ]
    if len(parameters) != 200:
        raise RuntimeError(
            f"expected 200 parameter tensors, got {len(parameters)}"
        )
    if sum(p.size for p in parameters) != 86_567_656:
        raise RuntimeError("captured parameter count disagrees with ViT-B/16")
    if args.import_dir is not None:
        records = json.loads((args.import_dir / "params.json").read_text())
        if len(records) != len(parameters):
            raise RuntimeError(
                "captured parameter order has changed since import"
            )
        for index, (record, value) in enumerate(zip(records, parameters)):
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            if (
                record["index"] != index
                or record["shape"] != list(value.shape)
                or record["sha256"] != digest
            ):
                raise RuntimeError(
                    f"captured parameter {index} differs from import/params.json"
                )
    actual_inputs = [pixels_np, *parameters]
    expected_shapes = subgraph_argument_shapes(subgraph_source, entry)
    if len(expected_shapes) != len(actual_inputs):
        raise RuntimeError(
            f"saved subgraph @{entry} expects {len(expected_shapes)} tensor inputs, "
            f"but the captured model supplies {len(actual_inputs)}; "
            "regenerate the replacement graph with the same capture ABI"
        )
    for index, (shape, value) in enumerate(zip(expected_shapes, actual_inputs)):
        if shape != value.shape or value.dtype != np.float32:
            raise RuntimeError(
                f"saved subgraph @{entry} argument {index} expects f32{shape}, "
                f"but captured input is {value.dtype}{value.shape}"
            )
    # Build Buddy's output descriptor from the captured graph, then replace its
    # high-level module with the saved Triton-call graph in the same context.
    graph.lower_to_top_level_ir()
    with ir.Location.unknown(graph._ctx):
        graph._imported_module = ir.Module.parse(
            subgraph_source, context=graph._ctx
        )
    started = time.time()
    graph.lower_to_llvm_ir(enable_openmp=False)
    lowering_seconds = round(time.time() - started, 3)
    llvm_lib = args.llvm_lib_dir
    libs = [
        llvm_lib / "libmlir_runner_utils.so",
        llvm_lib / "libmlir_c_runner_utils.so",
        args.host_library,
    ]
    require_paths(libs)
    engine = ExecutionEngine(
        graph._imported_module,
        opt_level=args.opt_level,
        shared_libs=[str(path) for path in libs],
    )
    keepalive = actual_inputs
    input_slots = []
    for value in keepalive:
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
        "mode": "Buddy baseline"
        if args.subgraph is not None
        else "Buddy and Triton replacement",
        "subgraph": str(subgraph),
        "subgraph_sha256": hashlib.sha256(subgraph.read_bytes()).hexdigest(),
        "entry": entry,
        "replacement_status": (
            report["status"] if args.subgraph is None else None
        ),
        "external_calls": (
            report["graphs"]["forward"]["external_calls"]
            if args.subgraph is None
            else 0
        ),
        "parameter_order_verified": args.import_dir is not None,
        "checkpoint_sha256": weights["checkpoint_sha256"],
        "input_shape": list(pixels_np.shape),
        "output_shape": list(result.shape),
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "mismatched_elements": int(np.count_nonzero(delta > tolerance)),
        "reference_top1": int(reference.argmax()),
        "compiled_top1": int(result.argmax()),
        "reference_first_logits": reference.reshape(-1)[:8].tolist(),
        "compiled_first_logits": result.reshape(-1)[:8].tolist(),
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
