#!/usr/bin/env python3
"""Bit-exact emitted-IR checks for independent tile-preparation candidates."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from check_split import command, prepare, small_case, export_kernel, Machine  # noqa: E402
from build_tile_opt import candidate_stages  # noqa: E402
from build_profile import stage_cases  # noqa: E402
from quantized_conv_oracle import patch_rows, quantize_rows  # noqa: E402
import tile_prototype  # noqa: E402


def emit(function, item, output):
    export_kernel(
        function,
        item["signature"],
        item["constexprs"],
        item["grid"],
        item["symbol"],
        output,
    )
    previous = output / "kernel.frontend.mlir"
    for filename, flags in [
        ("bufferized.mlir", ["--kernel-contract-verify"]),
        ("planned.mlir", ["--bosc-ame-plan", "--bosc-ame-plan-verify"]),
        ("boscame.mlir", ["--bosc-ame-emit"]),
    ]:
        target = output / filename
        command([os.environ["BUDDY_OPT"], previous, *flags, "-o", target])
        previous = target
    prepare(previous, output / "loops.mlir")
    machine = Machine(output / "loops.mlir", item["symbol"])
    for bank in machine.spm:
        bank[:] = bytes([0xD3]) * len(bank)
    return machine


def check(case, output):
    base = stage_cases(case)[1]
    spec = importlib.util.spec_from_file_location(
        "tile_check_baseline",
        ROOT / "build/w8a8-opt/baseline/quantized_conv_kernels.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = {
        "baseline": base,
        "batched": candidate_stages(case, "batched")[1],
        **{
            item["stage"]: item
            for item in candidate_stages(case, "vector_split")
            if item["stage"] in ("pure_im2col", "activation_quantize")
        },
    }
    entries["narrow"] = candidate_stages(case, "vector_narrow")[2]
    entries["vector_padding"] = candidate_stages(case, "vector_padding")[1]
    entries["cached_scalar"] = candidate_stages(case, "cached_scalar")[1]
    machines = {}
    for name, item in entries.items():
        function = (
            getattr(module, base["kernel"])
            if name == "baseline"
            else getattr(tile_prototype, item["kernel"])
        )
        machines[name] = emit(function, item, output / name)
    g = base["constexprs"]
    length = g["C"] * g["H"] * g["W"]
    xarray = (
        (np.arange(length) % 29 - 14).astype(np.float32) / np.float32(16)
    ).reshape(1, g["C"], g["H"], g["W"])
    x = bytearray(xarray.tobytes())
    patch = bytearray(4096 * 4)
    qb, qa, qv = bytearray(4096), bytearray(4096), bytearray(4096)
    sb, sa, sv = bytearray(256), bytearray(256), bytearray(256)
    qn, sn = bytearray(4096), bytearray(256)
    qp, sp = bytearray(4096), bytearray(256)
    qc, sc = bytearray(4096), bytearray(256)
    groups = g["C"] if g["DEPTHWISE"] else 1
    blocks = (g["OH"] * g["OW"] + 63) // 64
    arrays = {
        "X": xarray,
        "Weight": np.zeros(
            (g["C"], 1, g["KH"], g["KW"])
            if g["DEPTHWISE"]
            else (case["constexprs"]["OC"], g["C"], g["KH"], g["KW"]),
            np.float32,
        ),
        "Bias": np.zeros(
            g["C"] if g["DEPTHWISE"] else case["constexprs"]["OC"], np.float32
        ),
    }

    def launch(name, arrays, point):
        item = entries[name]
        buffers = {"%arg" + str(i): memoryview(array) for i, array in enumerate(arrays)}
        scalars = {
            "%arg" + str(len(arrays) + i): v
            for i, v in enumerate([*item["grid"], *point])
        }
        machines[name].launch(buffers, scalars)

    for group in range(groups):
        for block in range(blocks):
            for lane in range(64):
                launch("baseline", [x, qb, sb], [block * 64 + lane, 0, group])
            launch("batched", [x, qa, sa], [block, 0, group])
            launch("vector_padding", [x, qp, sp], [block, 0, group])
            launch("cached_scalar", [x, qc, sc], [block, 0, group])
            launch("pure_im2col", [x, patch], [block, 0, group])
            launch("activation_quantize", [patch, qv, sv], [block, 0, group])
            launch("narrow", [patch, qn, sn], [block, 0, group])
            fp = np.frombuffer(patch, dtype=np.float32).reshape(64, 64)
            expected_patch = patch_rows(case, arrays, group, block * 64)
            oq, oscale = quantize_rows(expected_patch, 64)
            assert np.array_equal(fp.view(np.uint32), expected_patch.view(np.uint32)), (
                "gather"
            )
            assert qb == qa == qv == qn == qp == qc == oq.tobytes(), (
                case["name"],
                group,
                block,
                "qpatch",
            )
            assert sb == sa == sv == sn == sp == sc == oscale.tobytes(), (
                case["name"],
                group,
                block,
                "scales",
            )
    return {
        "case": case["name"],
        "geometry": case["constexprs"],
        "status": "PASS",
        "tiles": groups * blocks,
        "padded_patch_rows": groups * blocks * 64,
        "coverage": "full small geometry with image-boundary and output-tail padding",
        "batched_bit_exact": True,
        "vector_split_bit_exact": True,
        "independent_oracle_bit_exact": True,
        "source_sha256": hashlib.sha256(
            (HERE / "tile_prototype.py").read_bytes()
        ).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "build/w8a8-opt/numeric")
    a = p.parse_args()
    cases = [
        small_case("conv2d_288x288_3x40_k3_s2", 7, 9, 3),
        small_case("dwconv2d_72x72x192_k3_s1", 7, 9, 3),
        small_case("dwconv2d_18x18x576_k5_s1", 7, 9, 2),
    ]
    results = []
    for case in cases:
        result = check(case, a.output / case["name"])
        results.append(result)
        print("PASS", case["name"], result["padded_patch_rows"], flush=True)
    (a.output / "candidate-consistency.json").write_text(
        json.dumps(results, indent=2) + chr(10)
    )


if __name__ == "__main__":
    main()
