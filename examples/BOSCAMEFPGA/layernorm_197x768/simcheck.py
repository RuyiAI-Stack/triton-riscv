#!/usr/bin/env python3
"""Check all 197 programs against the padded TTIR f32 semantics, without a board."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import build as example  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "tools"))
from paths import buddy_tools  # noqa: E402

sys.path.insert(0, str(buddy_tools()))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def oracle(x, weight, bias):
    padded = np.pad(x, (0, example.BLOCK - example.WIDTH))
    mean = np.float32(
        np.cumsum(padded, dtype=np.float32)[-1] / np.float32(example.WIDTH)
    )
    centered = np.subtract(padded, mean, dtype=np.float32)
    centered[example.WIDTH :] = np.float32(0)
    squared = np.multiply(centered, centered, dtype=np.float32)
    variance = np.float32(
        np.cumsum(squared, dtype=np.float32)[-1] / np.float32(example.WIDTH)
    )
    shifted = np.float32(variance + np.float32(example.EPS))
    root = np.sqrt(shifted, dtype=np.float32)
    inverse = np.float32(np.float32(1.0) / root)
    return np.add(
        np.multiply(
            np.multiply(centered[: example.WIDTH], inverse, dtype=np.float32),
            weight,
            dtype=np.float32,
        ),
        bias,
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module", type=Path, default=example.OUT / "kernel.boscame.mlir"
    )
    parser.add_argument(
        "--write-output",
        type=Path,
        help="Save all 197x768 output f32 bit patterns as raw bytes (no guards)",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        help="Require all output bytes to match a previous --write-output run",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        loops = Path(temp) / "kernel.loops.mlir"
        subprocess.run(
            [
                buddy_opt_path(),
                str(args.module),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(loops),
            ],
            check=True,
        )
        machine = Machine(loops, example.SYMBOL)
    rng = np.random.default_rng(71)
    x = rng.normal(0.8, 1.5, (example.ROWS, example.WIDTH)).astype(np.float32)
    weight = rng.uniform(-1, 2, example.WIDTH).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, example.WIDTH).astype(np.float32)
    guard = bytes([0xC7]) * 1024
    buffers = {
        "%arg0": bytearray(x.tobytes() + guard),
        "%arg1": bytearray(weight.tobytes() + guard),
        "%arg2": bytearray(bias.tobytes() + guard),
        "%arg3": bytearray(bytes(x.nbytes) + guard),
    }
    original = {
        name: bytes(data) for name, data in buffers.items() if name != "%arg3"
    }
    max_error = 0.0
    for row in reversed(range(example.ROWS)):
        start, end = row * example.WIDTH * 4, (row + 1) * example.WIDTH * 4
        machine.launch(
            buffers,
            {"%arg4": example.ROWS, "%arg5": 1, "%arg6": 1, "%arg7": row},
        )
        actual = np.frombuffer(buffers["%arg3"][start:end], dtype=np.float32)
        expected = oracle(x[row], weight, bias)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2.0e-5,
            atol=2.0e-5,
            err_msg=f"row {row}: output exceeds vector-reduction tolerance",
        )
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        assert machine.accesses[-4:] == [
            ("%arg0", start, end, "read"),
            ("%arg1", 0, example.WIDTH * 4, "read"),
            ("%arg2", 0, example.WIDTH * 4, "read"),
            ("%arg3", start, end, "write"),
        ]
    assert all(
        bytes(buffers[name]) == snapshot for name, snapshot in original.items()
    )
    assert buffers["%arg3"][-len(guard) :] == guard
    output = bytes(buffers["%arg3"][: x.nbytes])
    if args.compare_output is not None:
        golden = args.compare_output.read_bytes()
        if len(golden) != len(output):
            raise AssertionError(
                f"output byte count differs: got {len(output)}, "
                f"golden has {len(golden)}"
            )
        if output != golden:
            first = next(
                index
                for index, (actual, expected) in enumerate(zip(output, golden))
                if actual != expected
            )
            raise AssertionError(f"output bytes differ at byte offset {first}")
        print(f"OUTPUT CHECK PASS: all {len(output)} bytes match the golden")
    if args.write_output is not None:
        args.write_output.write_bytes(output)
    print(
        f"SIM CHECK PASS: {example.ROWS} programs, masked-lane f32 oracle rtol=2e-5, atol=2e-5, four exact 3072 B DMA ranges per row, DDR guards intact; max absolute error {max_error:.3g}"
    )


if __name__ == "__main__":
    main()
