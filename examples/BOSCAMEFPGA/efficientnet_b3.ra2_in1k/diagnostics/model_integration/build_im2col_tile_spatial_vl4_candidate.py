#!/usr/bin/env python3
"""Build the isolated K3/S1 full-tile snapshot plus spatial RVV candidate, retaining other stages."""

import argparse
import ctypes
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from cases import describe  # noqa: E402
from quantized_conv_oracle import patch_rows, quantize_rows  # noqa: E402
from build_conv_im2col_phase_candidate import function_bytes  # noqa: E402
from build_model_image import run, sha  # noqa: E402


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        help="optional real-node NPZ with input_0, weight and bias",
    )
    parser.add_argument(
        "--source", type=Path, default=HERE / "im2col_tile_spatial_vl4_candidate.c"
    )
    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="check every tile at the selected channel groups",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    case = describe(args.case)
    if case["family"] not in ("conv2d_f32", "dwconv2d_f32"):
        raise ValueError("requires one of the 18 W8A8 convolution cases")
    spec = case["constexprs"]
    depthwise = case["family"] == "dwconv2d_f32"
    if not depthwise or any(
        spec[key] != value
        for key, value in [("KH", 3), ("KW", 3), ("SH", 1), ("SW", 1)]
    ):
        raise ValueError("this isolated candidate requires depthwise K3/S1")
    source = args.source.resolve()
    raw = "triton_" + args.case + "__im2col_quantize"
    keys = ("H", "W", "C", "KH", "KW", "SH", "SW", "PH", "PW", "OH", "OW")
    definitions = ["-D" + key + "=" + str(spec[key]) for key in keys]
    definitions += ["-DDEPTHWISE=" + str(int(depthwise)), "-DIM2COL_ENTRY=" + raw]
    clang = REPO / "buddy-mlir/llvm/build/bin/clang"
    optimization = ["-O2", "-fno-vectorize", "-fno-slp-vectorize", "-ffp-contract=off"]
    host = args.output / "host.so"
    run(
        [clang, "-shared", "-fPIC", *optimization, *definitions, source, "-o", host],
        args.output / "host-build.log",
    )
    function = getattr(ctypes.CDLL(str(host)), raw)
    function.argtypes = [ctypes.c_int64, ctypes.c_void_p] * 3 + [ctypes.c_int32] * 6
    rng = np.random.default_rng(20261104)
    channels = spec["C"]
    oc = channels if depthwise else spec["OC"]
    weight_shape = (oc, 1 if depthwise else channels, spec["KH"], spec["KW"])
    arrays = {
        "X": rng.uniform(-11, 11, (1, channels, spec["H"], spec["W"])).astype(
            np.float32
        ),
        "Weight": np.zeros(weight_shape, np.float32),
        "Bias": np.zeros(oc, np.float32),
    }
    if args.fixture:
        with np.load(args.fixture) as fixture:
            arrays = {
                name: np.ascontiguousarray(fixture[key])
                for name, key in (
                    ("X", "input_0"),
                    ("Weight", "weight"),
                    ("Bias", "bias"),
                )
            }
    tiles = sorted(
        {0, (spec["OH"] * spec["OW"] // 64) // 2, (spec["OH"] * spec["OW"] - 1) // 64}
    )
    groups = sorted({0, channels // 2, channels - 1}) if depthwise else [0]
    if args.fixture or args.all_tiles:
        tiles = list(range((spec["OH"] * spec["OW"] + 63) // 64))
    checks = []
    for group in groups:
        for tile in tiles:
            q_guard = np.full(4096 + 128, 0x5D, np.uint8)
            s_guard = np.full(64 + 32, 0x7FA12345, np.uint32)
            q = q_guard[64:-64].view(np.int8).reshape(64, 64)
            scales = s_guard[16:-16].view(np.float32)
            # Production adapter initializes immutable padding once.
            q.fill(0)
            descriptors = [
                Descriptor(a.ctypes.data, a.ctypes.data, 0, a.size, 1)
                for a in (arrays["X"], q, scales)
            ]
            arguments = [value for d in descriptors for value in (1, ctypes.byref(d))]
            function(*arguments, len(tiles), 1, len(groups), tile, 0, group)
            patches = patch_rows(case, arrays, group, tile * 64)
            logical_k = spec["KH"] * spec["KW"] * (1 if depthwise else channels)
            expected_q, expected_scale = quantize_rows(patches[:, :logical_k])
            assert np.array_equal(q, expected_q), (group, tile, "QPatch")
            assert np.array_equal(
                scales.view(np.uint32), expected_scale.view(np.uint32)
            ), (group, tile, "scale")
            assert np.all(q_guard[:64] == 0x5D) and np.all(q_guard[-64:] == 0x5D)
            assert np.all(s_guard[:16] == 0x7FA12345) and np.all(
                s_guard[-16:] == 0x7FA12345
            )
            checks.append(
                {
                    "group": group,
                    "tile": tile,
                    "q_elements": 4096,
                    "scale_elements": 64,
                    "bit_exact": True,
                    "guards": True,
                }
            )
    target = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdcv_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-ffreestanding",
        "-fno-builtin",
    ]
    run(
        [
            clang,
            *target,
            *optimization,
            *definitions,
            "-S",
            source,
            "-o",
            args.output / "raw.s",
        ],
        args.output / "target-build.log",
    )
    native = (args.output / "raw.s").read_text().replace(".L", ".Ltilephase_")
    assembly = (args.kernel_dir / "kernel.nr.s").read_text()
    pattern = (
        r"^\s*\.globl\s+"
        + re.escape(raw)
        + r"\b[\s\S]*?^\s*\.size\s+"
        + re.escape(raw)
        + r",[^\n]*\n"
    )
    without, count = re.subn(pattern, "", assembly, count=1, flags=re.M)
    if count != 1:
        raise ValueError("could not isolate the original raw im2col function")
    restrict = REPO / "buddy-mlir/examples/FPGA-BOSCAME/tools/restrict_fpga_assembly.py"
    native_process = subprocess.run(
        [
            sys.executable,
            str(restrict),
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
            "--reject-vector-ordered-reductions",
            "--reject-vector-max-reductions",
        ],
        input=native,
        text=True,
        capture_output=True,
    )
    (args.output / "native-fences.log").write_text(native_process.stderr)
    if native_process.returncode:
        raise ValueError("native candidate ISA/fence restriction failed")
    native = native_process.stdout
    (args.output / "native-fenced.s").write_text(native)
    composite = without + "\n" + native
    process = subprocess.run(
        [
            sys.executable,
            str(restrict),
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
            "--reject-vector-ordered-reductions",
            "--reject-vector-max-reductions",
            "--no-fences",
        ],
        input=composite,
        text=True,
        capture_output=True,
    )
    (args.output / "restrict.log").write_text(process.stderr)
    if process.returncode:
        raise ValueError("candidate ISA restriction failed")
    (args.output / "kernel.nr.s").write_text(process.stdout)
    run(
        [
            clang,
            *target,
            "-c",
            args.output / "kernel.nr.s",
            "-o",
            args.output / "kernel.o",
        ],
        args.output / "object-build.log",
    )
    raw_process = subprocess.run(
        [
            sys.executable,
            str(restrict),
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
            "--reject-vector-ordered-reductions",
            "--reject-vector-max-reductions",
            "--no-fences",
        ],
        input=native,
        text=True,
        capture_output=True,
    )
    (args.output / "raw-restrict.log").write_text(raw_process.stderr)
    if raw_process.returncode:
        raise ValueError("raw candidate ISA restriction failed")
    (args.output / "raw-im2col.nr.s").write_text(raw_process.stdout)
    run(
        [
            clang,
            *target,
            "-c",
            args.output / "raw-im2col.nr.s",
            "-o",
            args.output / "raw-im2col.o",
        ],
        args.output / "raw-object-build.log",
    )
    before = function_bytes(args.kernel_dir / "kernel.o")
    after = function_bytes(args.output / "kernel.o")
    unchanged = {
        name: after[name] == value for name, value in before.items() if name != raw
    }
    if len(unchanged) != 7 or not all(unchanged.values()):
        raise ValueError("a weight/MMA/dequant/ABI function changed")
    for filename in ("adapter.c", "kernel.boscame.mlir"):
        (args.output / filename).write_bytes((args.kernel_dir / filename).read_bytes())
    report = {
        "status": "PASS_HOST_AND_BUILD_ONLY",
        "board_status": "NOT_RUN",
        "case": args.case,
        "source_sha256": sha(source),
        "source_path": str(source),
        "kernel_sha256": sha(args.output / "kernel.o"),
        "adapter_sha256": sha(args.output / "adapter.c"),
        "baseline_kernel_sha256": sha(args.kernel_dir / "kernel.o"),
        "raw_im2col_object_sha256": sha(args.output / "raw-im2col.o"),
        "fixture_sha256": sha(args.fixture) if args.fixture else None,
        "unchanged_non_im2col_functions": unchanged,
        "host_checks": checks,
        "source_math": "finite IEEE f32 max/127, div, truncation and ties away, unchanged quantization order",
        "protocol": "READ CPU gather 64x9 tap-major float stack; WRITE RVV stack-read math and QPatch[0:576]/scales; READ full CPU-owned byte snapshot; WRITE clear576 and scatter64x9; final drain",
        "vector_reductions": False,
        "strided_vector_memory": False,
        "temporary_qpatch_byte_range": [0, 576],
        "native_memory_fences_inserted_before_merge": True,
        "production_modified": False,
    }
    (args.output / "candidate-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "host_checks"},
            indent=2,
        )
    )
    print("host_checks", len(checks))


if __name__ == "__main__":
    main()
