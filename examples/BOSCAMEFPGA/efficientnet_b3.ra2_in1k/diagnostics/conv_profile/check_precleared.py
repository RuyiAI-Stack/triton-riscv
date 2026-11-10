#!/usr/bin/env python3
"""Actual emitted-IR QPatch equivalence under explicit one-time-zero contract."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import tile_prototype  # noqa: E402
from build_profile import stage_cases  # noqa: E402
from build_tile_opt import candidate_stages  # noqa: E402
from check_split import small_case  # noqa: E402
from check_tile_opt import emit  # noqa: E402
from export import inventory  # noqa: E402
from quantized_conv_oracle import patch_rows, quantize_rows  # noqa: E402


def check(case, output, sampled=False):
    item = stage_cases(case)[1]
    spec = importlib.util.spec_from_file_location(
        "precleared_baseline", ROOT / "quantized_conv_kernels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = candidate_stages(case, "cached_precleared")[1]
    machines = [
        emit(getattr(module, item["kernel"]), item, output / "current"),
        emit(
            tile_prototype.cached_precleared_tile_im2col_quantize,
            candidate,
            output / "candidate",
        ),
    ]
    g = item["constexprs"]
    shape = (1, g["C"], g["H"], g["W"])
    x = (
        (np.arange(np.prod(shape)) % 29 - 14).astype(np.float32) / np.float32(16)
    ).reshape(shape)
    xb = bytearray(x.tobytes())
    q = [bytearray(4096), bytearray(4096)]
    s = [bytearray(256), bytearray(256)]
    groups = g["C"] if g["DEPTHWISE"] else 1
    blocks = (g["OH"] * g["OW"] + 63) // 64
    arrays = {
        "X": x,
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
    selected_groups = sorted({0, groups // 2, groups - 1}) if sampled else range(groups)
    selected_blocks = sorted({0, blocks // 2, blocks - 1}) if sampled else range(blocks)
    for group in selected_groups:
        for block in selected_blocks:
            for i in range(2):
                buffers = {
                    "%arg0": memoryview(xb),
                    "%arg1": memoryview(q[i]),
                    "%arg2": memoryview(s[i]),
                }
                scalars = {
                    "%arg" + str(3 + j): value
                    for j, value in enumerate([*item["grid"], block, 0, group])
                }
                machines[i].launch(buffers, scalars)
            expected = patch_rows(case, arrays, group, block * 64)
            oq, os = quantize_rows(expected, 64)
            assert q[0] == q[1] == oq.tobytes(), (case["name"], group, block, "qpatch")
            assert s[0] == s[1] == os.tobytes(), (case["name"], group, block, "scale")
    return {
        "case": case["name"],
        "status": "PASS",
        "geometry": case["constexprs"],
        "tiles": len(selected_groups) * len(selected_blocks),
        "padded_patch_rows": len(selected_groups) * len(selected_blocks) * 64,
        "coverage": "first/middle/last original group/spatial tiles"
        if sampled
        else "full small group/spatial grid",
        "bit_exact": True,
        "contract": "adapter clears full QPatch exactly once before all group/block reuse; raw candidate writes only logical K",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-real",
        action="store_true",
        help="all 18 production geometries, first/middle/last group and spatial tiles",
    )
    args = parser.parse_args()
    cases = [
        small_case("conv2d_288x288_3x40_k3_s2", 7, 9, 3),
        small_case("dwconv2d_72x72x192_k3_s1", 7, 9, 3),
        small_case("dwconv2d_18x18x576_k5_s1", 7, 9, 2),
    ]
    if args.all_real:
        cases = [
            case
            for case in inventory()
            if case["family"] in ("conv2d_f32", "dwconv2d_f32")
        ]
    results = []
    for case in cases:
        result = check(case, args.output / case["name"], args.all_real)
        results.append(result)
        print("PASS", case["name"], result["padded_patch_rows"], flush=True)
    (args.output / "candidate-consistency.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
