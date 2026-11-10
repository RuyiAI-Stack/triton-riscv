#!/usr/bin/env python3
"""Strict BOSCAME small-grid equivalence of diagnostic split versus fused stages."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
from build_profile import compile_split, split_stage_cases, stage_cases  # noqa: E402
from export import describe, tool  # noqa: E402
from quantized_conv_oracle import patch_rows, quantize_rows  # noqa: E402

sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/tools"))
from contract_frontend import export_kernel  # noqa: E402

sys.path.insert(0, str(REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402


def command(args):
    result = subprocess.run(list(map(str, args)), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def prepare(source, destination):
    command(
        [
            tool("buddy-opt"),
            source,
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
            "-o",
            destination,
        ]
    )


def small_case(name, width, height, channels):
    case = copy.deepcopy(describe(name))
    spec = case["constexprs"]
    spec.update(H=height, W=width, C=channels)
    spec["OH"] = (height + 2 * spec["PH"] - spec["KH"]) // spec["SH"] + 1
    spec["OW"] = (width + 2 * spec["PW"] - spec["KW"]) // spec["SW"] + 1
    return case


def check(case, output, selected_real_tiles=False):
    output.mkdir(parents=True, exist_ok=True)
    compile_split(case, output)
    prepare(output / "kernel.boscame.mlir", output / "split.loops.mlir")
    fused = next(
        item for item in stage_cases(case) if item["stage"] == "im2col_quantize"
    )
    spec = importlib.util.spec_from_file_location(
        "diag_prod", ROOT / "quantized_conv_kernels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fused_dir = output / "fused"
    fused_dir.mkdir(exist_ok=True)
    export_kernel(
        getattr(module, fused["kernel"]),
        fused["signature"],
        fused["constexprs"],
        fused["grid"],
        fused["symbol"],
        fused_dir,
    )
    previous = fused_dir / "kernel.frontend.mlir"
    for filename, flags in [
        ("bufferized.mlir", ["--kernel-contract-verify"]),
        ("planned.mlir", ["--bosc-ame-plan", "--bosc-ame-plan-verify"]),
        ("boscame.mlir", ["--bosc-ame-emit"]),
    ]:
        destination = fused_dir / filename
        command([tool("buddy-opt"), previous, *flags, "-o", destination])
        previous = destination
    prepare(previous, output / "fused.loops.mlir")
    stages = {item["stage"]: item for item in split_stage_cases(case)}
    machines = {
        name: Machine(output / "split.loops.mlir", item["symbol"])
        for name, item in stages.items()
        if name in ("pure_im2col", "activation_quantize")
    }
    machines["im2col_quantize"] = Machine(output / "fused.loops.mlir", fused["symbol"])
    for machine in machines.values():
        for bank in machine.spm:
            bank[:] = bytes([0xD3]) * len(bank)
    geometry = fused["constexprs"]
    length = geometry["C"] * geometry["H"] * geometry["W"]
    values = (
        (np.arange(length) % 29 - 14).astype(np.float32) / np.float32(16)
    ).reshape(1, geometry["C"], geometry["H"], geometry["W"])
    x = bytearray(values.tobytes())
    patch = bytearray(4096 * 4)
    qsplit, qfused = bytearray(4096), bytearray(4096)
    ssplit, sfused = bytearray(64 * 4), bytearray(64 * 4)

    def launch(name, arrays, point):
        item = fused if name == "im2col_quantize" else stages[name]
        buffers = {"%arg" + str(i): memoryview(array) for i, array in enumerate(arrays)}
        scalars = {
            "%arg" + str(len(arrays) + i): value
            for i, value in enumerate([*item["grid"], *point])
        }
        machines[name].launch(buffers, scalars)

    positions = geometry["OH"] * geometry["OW"]
    tile = geometry["TILE"]
    blocks = (positions + tile - 1) // tile
    groups = geometry["C"] if geometry["DEPTHWISE"] else 1
    assert fused["grid"] == [blocks, 1, groups], fused["grid"]
    block_ids = (
        sorted({0, blocks // 2, blocks - 1})
        if selected_real_tiles
        else list(range(blocks))
    )
    group_ids = (
        sorted({0, groups // 2, groups - 1})
        if selected_real_tiles
        else list(range(groups))
    )
    channels = geometry["C"] if geometry["DEPTHWISE"] else case["constexprs"]["OC"]
    weight_shape = (
        channels,
        1 if geometry["DEPTHWISE"] else geometry["C"],
        geometry["KH"],
        geometry["KW"],
    )
    arrays = {
        "X": values,
        "Weight": np.zeros(weight_shape, dtype=np.float32),
        "Bias": np.zeros(channels, dtype=np.float32),
    }
    points = []
    for group in group_ids:
        for block in block_ids:
            # The adapter invokes one raw program per TILE-row patch tile.
            # Dirty every split output so missing lanes/padding cannot pass by reuse.
            patch[:] = bytes([0xD3]) * len(patch)
            qsplit[:] = bytes([0xD3]) * len(qsplit)
            ssplit[:] = bytes([0xD3]) * len(ssplit)
            launch("pure_im2col", [x, patch], [block, 0, group])
            launch("activation_quantize", [patch, qsplit, ssplit], [block, 0, group])
            # Production QPatch padding was cleared once above, as in its adapter.
            launch("im2col_quantize", [x, qfused, sfused], [block, 0, group])
            if qsplit != qfused:
                raise AssertionError(f"QPatch mismatch group={group} block={block}")
            if ssplit != sfused:
                raise AssertionError(f"PatchScale mismatch group={group} block={block}")
            gathered = patch_rows(case, arrays, group, block * tile, tile)
            if patch != gathered.tobytes():
                raise AssertionError(
                    f"FPatch oracle mismatch group={group} block={block}"
                )
            oracle_q, oracle_s = quantize_rows(gathered, 64)
            if oracle_q.tobytes() != qsplit or oracle_s.tobytes() != ssplit:
                raise AssertionError(
                    f"quantization oracle mismatch group={group} block={block}"
                )
            points.append(
                {"group": group, "block": block, "position_start": block * tile}
            )
    return {
        "case": case["name"],
        "geometry": case["constexprs"],
        "status": "PASS",
        "coverage": "first/middle/last real tiles and groups"
        if selected_real_tiles
        else "small geometry full tile grid, all lanes, groups, padding and tail",
        "raw_grid": fused["grid"],
        "tile_programs": len(points),
        "patch_rows": len(points) * tile,
        "tested_points": points,
        "qpatch_bit_exact": True,
        "patch_scale_bit_exact": True,
        "fpatch_oracle_bit_exact": True,
        "oracle_independent_of_split_output": True,
        "split_variant": "scalar_logical_width_tile_v2",
        "source_sha256": hashlib.sha256(
            (HERE / "split_kernels.py").read_bytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/w8a8-profile/split-check"
    )
    args = parser.parse_args()
    # Each small geometry has >= 3 tile programs and a partial final tile.
    cases = [
        small_case("conv2d_288x288_3x40_k3_s2", 31, 29, 3),
        small_case("dwconv2d_72x72x192_k3_s1", 17, 13, 3),
        small_case("dwconv2d_18x18x576_k5_s1", 17, 13, 2),
    ]
    results = []
    for case in cases:
        result = check(case, args.output / "small" / case["name"])
        results.append(result)
        print(
            "PASS small",
            case["name"],
            "tile_programs=" + str(result["tile_programs"]),
            flush=True,
        )
    for name in ("conv2d_288x288_3x40_k3_s2", "dwconv2d_72x72x192_k3_s1"):
        result = check(
            describe(name),
            args.output / "real-selected" / name,
            selected_real_tiles=True,
        )
        results.append(result)
        print(
            "PASS real-selected",
            name,
            "tile_programs=" + str(result["tile_programs"]),
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split-fused-consistency.json").write_text(
        json.dumps(results, indent=2) + chr(10)
    )


if __name__ == "__main__":
    main()
