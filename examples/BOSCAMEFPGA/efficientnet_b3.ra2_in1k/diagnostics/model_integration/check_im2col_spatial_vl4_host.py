#!/usr/bin/env python3
"""Independent QPatch/scale/guard checks for the isolated spatial-lane candidate.

The host build emulates the vector batch's math and shared alias protocol. It
does not execute RVV instructions; target RVV correctness needs separate FPGA
output validation. Check all tiles at first/middle/last channel groups, padding
reused between raw calls, zero rows, half ties, row crossings and synthetic tails.
"""

import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from cases import describe  # noqa: E402
from quantized_conv_oracle import patch_rows, quantize_rows  # noqa: E402
from build_im2col_spatial_vl4_candidate import Descriptor, run, sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = json.loads((args.candidate_dir / "candidate-manifest.json").read_text())
    source = Path(candidate["source_path"])
    if sha(source) != candidate["source_sha256"]:
        raise ValueError("candidate source changed")
    original = describe(candidate["case"])
    clang = REPO / "buddy-mlir/llvm/build/bin/clang"
    geometries = [("full_72x72", original)]
    for h, w in [(9, 9), (5, 7), (2, 3)]:
        case = dict(original)
        case["constexprs"] = {
            **original["constexprs"],
            "H": h,
            "W": w,
            "C": 3,
            "OH": h,
            "OW": w,
        }
        geometries.append((f"synthetic_{h}x{w}", case))
    checks = []
    for geometry, case in geometries:
        spec = case["constexprs"]
        symbol = "host_spatial_vl4_" + geometry
        definitions = [
            "-D" + key + "=" + str(spec[key])
            for key in ("H", "W", "C", "KH", "KW", "SH", "SW", "PH", "PW", "OH", "OW")
        ]
        definitions += ["-DDEPTHWISE=1", "-DIM2COL_ENTRY=" + symbol]
        library = args.output / (geometry + ".so")
        run(
            [
                clang,
                "-shared",
                "-fPIC",
                "-O2",
                "-fno-vectorize",
                "-fno-slp-vectorize",
                "-ffp-contract=off",
                *definitions,
                source,
                "-o",
                library,
            ],
            args.output / (geometry + "-build.log"),
        )
        function = getattr(ctypes.CDLL(str(library.resolve())), symbol)
        function.argtypes = [ctypes.c_int64, ctypes.c_void_p] * 3 + [ctypes.c_int32] * 6
        shape = (1, spec["C"], spec["H"], spec["W"])
        rng = np.random.default_rng(20261104)
        random = rng.uniform(-11, 11, shape).astype(np.float32)
        zero = np.zeros(shape, np.float32)
        zero.reshape(-1)[::2] = np.float32(-0.0)
        boundaries = np.array(
            [
                -127.0,
                -126.5,
                -2.5,
                -1.5,
                -0.5,
                np.nextafter(np.float32(-0.5), np.float32(0)),
                -0.0,
                0.0,
                np.nextafter(np.float32(0.5), np.float32(0)),
                0.5,
                1.5,
                2.5,
                126.5,
                127.0,
            ],
            np.float32,
        )
        ties = np.resize(boundaries, np.prod(shape)).reshape(shape).copy()
        # Every three input rows contain scale=1 maxima for interior patches.
        ties[:, :, ::3, :] = np.float32(127)
        scenarios = [
            ("signed_random", random),
            ("signed_zero", zero),
            ("half_ties", ties),
        ]
        groups = sorted({0, spec["C"] // 2, spec["C"] - 1})
        tiles = (spec["OH"] * spec["OW"] + 63) // 64
        for scenario, x in scenarios:
            arrays = {
                "X": x,
                "Weight": np.zeros((spec["C"], 1, 3, 3), np.float32),
                "Bias": np.zeros(spec["C"], np.float32),
            }
            for group in groups:
                q_guard = np.full(4096 + 128, 0x5D, np.uint8)
                s_guard = np.full(64 + 32, 0x7FA12345, np.uint32)
                q = q_guard[64:-64].view(np.int8).reshape(64, 64)
                scales = s_guard[16:-16].view(np.float32)
                q.fill(0)  # Once for this sequence, matching public-call contract.
                for tile in range(tiles):
                    q[:, :9] = 0x23
                    scales.view(np.uint32).fill(0x7FC12345)
                    descriptors = [
                        Descriptor(a.ctypes.data, a.ctypes.data, 0, a.size, 1)
                        for a in (arrays["X"], q, scales)
                    ]
                    arguments = [v for d in descriptors for v in (1, ctypes.byref(d))]
                    function(*arguments, tiles, 1, spec["C"], tile, 0, group)
                    expected_q, expected_scale = quantize_rows(
                        patch_rows(case, arrays, group, tile * 64)[:, :9]
                    )
                    assert np.array_equal(q, expected_q), (
                        geometry,
                        scenario,
                        group,
                        tile,
                        "Q",
                    )
                    assert np.array_equal(
                        scales.view(np.uint32), expected_scale.view(np.uint32)
                    ), (geometry, scenario, group, tile, "scales")
                    assert np.all(q[:, 9:] == 0), "temporary padding not restored"
                    assert np.all(q_guard[:64] == 0x5D) and np.all(
                        q_guard[-64:] == 0x5D
                    )
                    assert np.all(s_guard[:16] == 0x7FA12345) and np.all(
                        s_guard[-16:] == 0x7FA12345
                    )
                    checks.append(
                        {
                            "geometry": geometry,
                            "scenario": scenario,
                            "group": group,
                            "tile": tile,
                            "q_elements": 4096,
                            "scales": 64,
                            "padding_zero": True,
                            "bit_exact": True,
                            "guards": True,
                        }
                    )
    report = {
        "status": "PASS_HOST_EMULATION_ONLY",
        "source_sha256": sha(source),
        "source": str(source),
        "kernel_sha256": candidate["kernel_sha256"],
        "scope": "host emulation of vector arithmetic plus identical alias protocol; not RVV execution",
        "calls": len(checks),
        "q_elements": sum(x["q_elements"] for x in checks),
        "scale_elements": sum(x["scales"] for x in checks),
        "checks": checks,
    }
    (args.output / "host-independent-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
