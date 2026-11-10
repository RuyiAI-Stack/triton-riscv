#!/usr/bin/env python3
"""Check independent W8A8 NCHW packing on all real model connector shapes."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
BUDDY = REPO / "buddy-mlir"
LLVM = BUDDY / "llvm/build/bin"
BASE = MODEL / "build/model-integration"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log, source=None):
    result = subprocess.run(
        list(map(str, command)),
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    log.write_text(repr(list(map(str, command))) + "\n" + result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("command failed: " + str(log))
    return result.stdout


def check_output(native, value, expected_q, expected_scale, label):
    value = np.ascontiguousarray(value, dtype=np.float32)
    channels, height, width = value.shape[1:]
    spatial = height * width
    q_guard = np.full(expected_q.size + 32, 0x5D, dtype=np.uint8)
    scale_guard = np.full(expected_scale.size + 32, 0x7FA12345, dtype=np.uint32)
    q = q_guard[16:-16].view(np.int8)
    scale = scale_guard[16:-16].view(np.float32)
    native(value.ctypes.data, q.ctypes.data, scale.ctypes.data, channels, spatial)
    if not np.array_equal(q, expected_q.reshape(-1)):
        raise AssertionError(label + ": int8 bytes differ from host oracle")
    if not np.array_equal(scale.view(np.uint32), expected_scale.view(np.uint32)):
        raise AssertionError(label + ": activation scales differ from host oracle")
    if not np.all(q_guard[:16] == 0x5D) or not np.all(q_guard[-16:] == 0x5D):
        raise AssertionError(label + ": packed output guard overwritten")
    if not np.all(scale_guard[:16] == 0x7FA12345) or not np.all(
        scale_guard[-16:] == 0x7FA12345
    ):
        raise AssertionError(label + ": scale output guard overwritten")
    return {
        "q_elements": int(q.size),
        "scale_elements": int(scale.size),
        "input_min": float(value.min()),
        "input_max": float(value.max()),
        "input_max_abs": float(np.max(np.abs(value))),
        "bit_exact": True,
        "guards_intact": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "quantize-rvv")
    parser.add_argument(
        "--plan", type=Path, default=BASE / "physical-plan/physical-plan.json"
    )
    parser.add_argument(
        "--graph", type=Path, default=BASE / "shape-graph-manifest.json"
    )
    parser.add_argument("--input-f32", type=Path, default=BASE / "reference/input.f32")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source = HERE / "quantize_rvv_candidate.c"
    library = args.output / "host.so"
    run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-fno-tree-vectorize",
            "-fno-tree-slp-vectorize",
            source,
            "-o",
            library,
        ],
        args.output / "host-build.log",
    )
    native = ctypes.CDLL(str(library.resolve())).efficientnet_quantize_pack_candidate
    native.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]

    sys.path.insert(0, str(BASE / "python-deps"))
    spec = importlib.util.spec_from_file_location(
        "quantize_reference", HERE / "reference.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch = reference.torch
    torch.set_num_threads(4)
    graph = json.loads(args.graph.read_text())
    plan = json.loads(args.plan.read_text())
    specs = {
        item["name"]: item
        for item in plan["connectors"]
        if item["kernel"] == "quantize_pack_nchw"
    }
    calls = {
        call["logical_node"]: call
        for call in plan["calls"]
        if call["target_kind"] == "connector"
        and call["target"].startswith("quantize_pack_nchw__")
    }
    wanted_nodes = set(calls)
    wanted = {
        (
            item["constexprs"]["C"],
            item["constexprs"]["H"],
            item["constexprs"]["W"],
        ): item["name"]
        for item in specs.values()
    }
    activations = {}
    original_conv = reference.w8a8_conv

    def capture_conv(value, module, parameters):
        _, channels, height, width = value.shape
        key = (channels, height, width)
        node = module_names.get(id(module))
        if node in wanted_nodes and key in wanted and key not in activations:
            activations[key] = value.detach().cpu().contiguous().numpy().copy()
        return original_conv(value, module, parameters)

    reference.w8a8_conv = capture_conv
    model, _ = reference.load_model(BASE / "assets")
    module_names = {
        id(dict(model.named_modules())[node["name"]]): node["name"]
        for node in graph["nodes"]
        if node["kind"] == "conv" and node["name"] in wanted_nodes
    }
    image = torch.from_numpy(
        np.fromfile(args.input_f32, dtype="<f4").reshape(1, 3, 288, 288).copy()
    )
    parameters = reference.folded_parameters(model, graph)
    with torch.inference_mode():
        reference.interpret(model, graph, image, parameters, True)
    missing = sorted(set(wanted) - set(activations))
    if missing:
        raise ValueError(
            "no real W8A8 activation matched connector shapes: " + repr(missing)
        )

    results = []
    for shape, name in sorted(wanted.items()):
        tensor = torch.from_numpy(activations[shape])
        rows = (
            tensor[0]
            .permute(1, 2, 0)
            .reshape(shape[1] * shape[2], shape[0])
            .contiguous()
        )
        expected_q_t, expected_scale_t = reference.quantize_rows(rows)
        expected_q = expected_q_t.numpy().astype(np.int8, copy=False)
        expected_scale = expected_scale_t.numpy().astype(np.float32, copy=False)
        result = check_output(
            native, activations[shape], expected_q, expected_scale, name
        )
        result.update(
            {
                "connector": name,
                "shape": [1, *shape],
                "input_sha256": hashlib.sha256(
                    activations[shape].tobytes()
                ).hexdigest(),
                "q_sha256": hashlib.sha256(expected_q.tobytes()).hexdigest(),
                "scale_sha256": hashlib.sha256(expected_scale.tobytes()).hexdigest(),
            }
        )
        results.append(result)

    # Exercise signed zero, zero rows, clipping and half-away ties separately.
    edge = np.zeros((1, 5, 3, 3), dtype=np.float32)
    edge[0, :, 0, 0] = [-0.0, 0.0, 0.0, -0.0, 0.0]
    edge[0, :, 0, 1] = [1.27, -0.635, 0.005, -0.015, 0.0]
    edge[0, :, 0, 2] = [3.0, -3.0, 2.0, -1.0, 0.0]
    rows = torch.from_numpy(edge[0].transpose(1, 2, 0).reshape(9, 5).copy())
    eq, es = reference.quantize_rows(rows)
    edge_result = check_output(
        native,
        edge,
        eq.numpy().astype(np.int8),
        es.numpy().astype(np.float32),
        "finite-boundaries",
    )

    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdcv_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
    ]
    assembly = args.output / "kernel.s"
    run(
        [LLVM / "clang", *flags, "-S", source, "-o", assembly],
        args.output / "compile.log",
    )
    tools = BUDDY / "examples/FPGA-BOSCAME/tools"
    restricted = run(
        [
            sys.executable,
            tools / "restrict_fpga_assembly.py",
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
            "--reject-vector-ordered-reductions",
            "--reject-vector-max-reductions",
        ],
        args.output / "restrict.log",
        assembly.read_text(),
    )
    restricted_path = args.output / "kernel.nr.s"
    restricted_path.write_text(restricted)
    obj = args.output / "kernel.o"
    run(
        [LLVM / "clang", *flags, "-c", restricted_path, "-o", obj],
        args.output / "object.log",
    )
    report = {
        "status": "LOCAL_PASS_BOARD_PENDING",
        "scope": "host real-weight connector inputs and finite edge fixtures; no FPGA evidence",
        "connectors_expected": len(wanted),
        "connectors_checked": len(results),
        "real_activation_shapes": results,
        "edge_fixture": edge_result,
        "candidate_source_sha256": sha(source),
        "host_library_sha256": sha(library),
        "candidate_object_sha256": sha(obj),
        "riscv_isa_restriction_pass": True,
        "production_changed": False,
        "fpga_numeric": "pending",
        "fpga_abab": "pending",
    }
    (args.output / "local-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "connectors_checked": len(results),
                "real_outputs": sum(r["q_elements"] for r in results),
                "edge_guards": edge_result["guards_intact"],
                "riscv_isa_restriction_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
