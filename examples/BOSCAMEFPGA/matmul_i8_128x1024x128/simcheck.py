#!/usr/bin/env python3
"""Bit-exact DDR-level check of the generated experimental BOSCAME kernel.

This script runs the experimental BOSCAME pass output:
every program of the launch grid is executed by `tools/boscame_buffer_sim.py` (the
functional NH/E6 model -- real DMA contract, two 128 KiB SPM banks, AME tile
semantics) and the resulting C is compared against an independent NumPy INT32
oracle. SPM persists between programs, and guards/access logs check excluded DDR.

    python3 simcheck.py [--module PATH] [--programs N]

The oracle and launch read the shape and grid from ``build.py``; this keeps
the simulator aligned with ``kernels.py``. Requires numpy.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
import build as example  # noqa: E402  (artifact path and entry symbol)

sys.path.insert(0, str(ROOT.parent / "tools"))
from paths import buddy_tools  # noqa: E402

sys.path.insert(0, str(buddy_tools()))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402


def operands():
    """The deterministic A, B and initial C the launcher fills (launch.c)."""
    m, n, k = (example.SHAPE[name] for name in ("M", "N", "K"))
    a = (np.arange(m)[:, None] * 13 + np.arange(k)[None, :] * 7) % 31 - 15
    b = (np.arange(n)[:, None] * 11 + np.arange(k)[None, :] * 5) % 29 - 14
    c = (np.arange(m)[:, None] * 3 + np.arange(n)[None, :]) % 7 - 3
    return a.astype(np.int8), b.astype(np.int8), c.astype("<i4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        type=Path,
        default=example.OUT / "kernel.boscame.mlir",
        help="pass output (default: build/<name>/kernel.boscame.mlir)",
    )
    parser.add_argument(
        "--programs",
        type=int,
        default=0,
        help="stop after N programs (default: the whole grid)",
    )
    args = parser.parse_args()
    if not args.module.is_file():
        raise SystemExit("missing " + str(args.module) + "; run `make boscame`")

    a, b, c0 = operands()
    guard = bytes([0xA7]) * 1024
    buffers = {
        "%arg0": bytearray(a.tobytes() + guard),
        "%arg1": bytearray(b.tobytes() + guard),
        "%arg2": bytearray(c0.tobytes() + guard),
    }
    # Parse only supported emitted operations; unknown forms fail closed.
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
    grid_m = example.SHAPE["M"] // example.SHAPE["BM"]
    grid_n = example.SHAPE["N"] // example.SHAPE["BN"]
    total = grid_m * grid_n
    limit = args.programs or total
    launched = 0
    for x in range(grid_m):
        for y in range(grid_n):
            for z in range(1):
                if launched >= limit:
                    break
                machine.launch(
                    buffers,
                    {
                        "%arg3": grid_m,
                        "%arg4": grid_n,
                        "%arg5": 1,
                        "%arg6": x,
                        "%arg7": y,
                        "%arg8": z,
                    },
                )
                launched += 1

    want = c0.astype(np.int64) + a.astype(np.int64) @ b.astype(np.int64).T
    got = np.frombuffer(
        bytes(buffers["%arg2"]),
        dtype="<i4",
        count=example.SHAPE["M"] * example.SHAPE["N"],
    ).reshape(example.SHAPE["M"], example.SHAPE["N"])
    assert buffers["%arg0"] == a.tobytes() + guard
    assert buffers["%arg1"] == b.tobytes() + guard
    assert buffers["%arg2"][-len(guard) :] == guard
    limits = {"%arg0": a.nbytes, "%arg1": b.nbytes, "%arg2": c0.nbytes}
    assert all(
        0 <= start < end <= limits[name]
        for name, start, end, _ in machine.accesses
    )
    max_abs = int(np.abs(got.astype(np.int64) - want).max())
    print(f"simulated {launched}/{total} programs, max_abs={max_abs}")
    if launched != total or max_abs:
        print("SIM CHECK FAIL")
        return 1
    print(
        "SIM CHECK PASS: "
        f"{example.SHAPE['M'] * example.SHAPE['N']} outputs match the NumPy "
        f"INT32 oracle after {total} programs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
