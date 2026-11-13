#!/usr/bin/env python3
"""Check independent batched dequantization semantics and the actual NR object."""

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1]
REPO = MODEL.parents[2]
BUDDY = REPO / "buddy-mlir"
LLVM = BUDDY / "llvm/build/bin"


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
        raise RuntimeError(str(log))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=MODEL / "build/model-integration/dequant-rvv"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = HERE / "dequant_rvv_candidate.c"
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
    native = ctypes.CDLL(str(library.resolve())).efficientnet_dequant_rvv_candidate
    native.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_size_t] * 2
    rng = np.random.default_rng(112)
    results = []
    for channels, spatial in (
        (1, 1),
        (3, 7),
        (10, 1),
        (24, 81),
        (144, 20736),
        (816, 324),
        (1392, 81),
    ):
        acc = rng.integers(-(2**31), 2**31, (spatial, channels), dtype=np.int32)
        act = rng.uniform(-1, 1, spatial).astype(np.float32)
        ws = rng.uniform(-1, 1, channels).astype(np.float32)
        bias = rng.uniform(-10, 10, channels).astype(np.float32)
        # Include signed zero, small scales and integer conversion boundaries.
        act[: min(4, spatial)] = np.array([0.0, -0.0, 2**-126, 2**-149], np.float32)[
            : min(4, spatial)
        ]
        acc.flat[: min(6, acc.size)] = [
            -2147483648,
            2147483647,
            -16777217,
            16777217,
            0,
            1,
        ][: min(6, acc.size)]
        guard = np.full(acc.size + 32, 0x513A7E29, dtype=np.uint32)
        out = guard[16:-16].view(np.float32).reshape(channels, spatial)
        expected = acc.astype(np.float32) * ws[None, :]
        expected = expected * act[:, None]
        expected = (expected + bias[None, :]).T.copy()
        native(
            acc.ctypes.data,
            act.ctypes.data,
            ws.ctypes.data,
            bias.ctypes.data,
            out.ctypes.data,
            channels,
            spatial,
        )
        assert np.array_equal(out.view(np.uint32), expected.view(np.uint32))
        assert np.all(guard[:16] == 0x513A7E29) and np.all(guard[-16:] == 0x513A7E29)
        results.append(
            {
                "channels": channels,
                "spatial": spatial,
                "elements": acc.size,
                "bit_exact": True,
            }
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
    (args.output / "kernel.nr.s").write_text(restricted)
    obj = args.output / "kernel.o"
    run(
        [LLVM / "clang", *flags, "-c", args.output / "kernel.nr.s", "-o", obj],
        args.output / "object.log",
    )
    report = {
        "status": "LOCAL_PASS_BOARD_PENDING",
        "candidate_source_sha256": sha(source),
        "host_library_sha256": sha(library),
        "candidate_object_sha256": sha(obj),
        "host_semantics": results,
        "host_bit_exact": True,
        "riscv_isa_restriction_pass": True,
        "fpga_numeric": "pending",
        "fpga_abab": "pending",
        "production_changed": False,
        "hypothesis": "one raw entry per tensor and two DDR policy transitions per <=16 outputs",
        "baseline": "five scalar DDR accesses plus one raw entry per output",
    }
    (args.output / "local-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
