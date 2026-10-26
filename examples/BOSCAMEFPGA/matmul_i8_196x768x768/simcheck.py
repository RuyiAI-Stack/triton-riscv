#!/usr/bin/env python3
"""Bit-exact DDR-level check of the generated experimental BOSCAME kernel.

This script runs the experimental BOSCAME pass output: every program of the
launch grid is executed by `tools/boscame_buffer_sim.py` (the functional NH/E6
model -- real DMA contract, two 128 KiB SPM banks, AME tile semantics) and the
resulting C is compared against an independent NumPy INT32 oracle.  SPM
persists between programs, and guards/access logs check excluded DDR.

The operands are the logical [196,768] / [768,768] buffers the launcher fills;
the last M program is the masked boundary tile (196 is not a multiple of 32).

    python3 simcheck.py [--module PATH] [--programs N]

The oracle and launch read the shape and grid from ``build.py``; this keeps the
simulator aligned with ``kernels.py``.  Requires numpy.
"""

import argparse
import hashlib
import itertools
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


def extreme_operands():
    """Asymmetric signed bytes and nonzero C that exercise INT32 wrap."""
    m, n, k = (example.SHAPE[name] for name in ("M", "N", "K"))
    rng = np.random.default_rng(923)
    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (n, k), dtype=np.int8)
    a[::2, ::2] = -128
    b[::3, 1::2] = 127
    c = rng.integers(-(1 << 31), 1 << 31, (m, n), dtype=np.int32)
    c[::2, ::2] = np.iinfo(np.int32).max
    c[1::2, 1::2] = np.iinfo(np.int32).min
    return a, b, c.astype("<i4")


def check_accesses(accesses, x, y):
    """Every DDR access stays inside this program's authorized rectangle."""
    m, n, k, bm, bn = (
        example.SHAPE[name] for name in ("M", "N", "K", "BM", "BN")
    )
    rows = min(bm, m - x * bm)
    counts = {}
    for name, start, end, kind in accesses:
        pitch = k if name != "%arg2" else n * 4
        row = start // pitch
        lo, hi = (
            (y * bn, (y + 1) * bn)
            if name == "%arg1"
            else (x * bm, x * bm + rows)
        )
        left, right = (
            (y * bn * 4, (y + 1) * bn * 4) if name == "%arg2" else (0, k)
        )
        assert (
            lo <= row < hi
            and row * pitch + left <= start < end <= row * pitch + right
        )
        assert name == "%arg2" or kind == "read"
        counts[name, kind] = counts.get((name, kind), 0) + end - start
    assert counts == {
        ("%arg0", "read"): rows * k,
        ("%arg1", "read"): bn * k,
        ("%arg2", "read"): rows * bn * 4,
        ("%arg2", "write"): rows * bn * 4,
    }, counts


def check_case(machine, data, order, limit, label):
    a, b, c0 = data
    guard = bytes([0xA7]) * 1024
    buffers = {
        "%arg0": bytearray(a.tobytes() + guard),
        "%arg1": bytearray(b.tobytes() + guard),
        "%arg2": bytearray(c0.tobytes() + guard),
    }
    grid_m, grid_n = (example.SHAPE[name] for name in ("GRID_M", "GRID_N"))
    for x, y, z in order[:limit]:
        first = len(machine.accesses)
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
        check_accesses(machine.accesses[first:], x, y)

    # The oracle accumulates in INT64, then independently applies INT32 wrap.
    want = (
        c0.astype(np.int64) + a.astype(np.int64) @ b.astype(np.int64).T
    ).astype("<i4")
    got = np.frombuffer(buffers["%arg2"], dtype="<i4", count=c0.size).reshape(
        c0.shape
    )
    max_abs = int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max())
    assert buffers["%arg0"] == a.tobytes() + guard
    assert buffers["%arg1"] == b.tobytes() + guard
    assert buffers["%arg2"][-len(guard) :] == guard
    output = got.tobytes()
    print(
        f"{label}: {min(limit, len(order))}/{len(order)} programs, max_abs={max_abs}, sha256={hashlib.sha256(output).hexdigest()}",
        flush=True,
    )
    assert limit >= len(order) and max_abs == 0, "SIM CHECK FAIL"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Also check signed extrema/INT32 wrap with tail-first and reverse launches",
    )
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
    grid_m = example.SHAPE["GRID_M"]
    grid_n = example.SHAPE["GRID_N"]
    total = grid_m * grid_n
    limit = args.programs or total
    order = list(itertools.product(range(grid_m), range(grid_n), range(1)))
    check_case(machine, operands(), order, limit, "launcher/forward")
    if args.stress:
        data = extreme_operands()
        # Each bank starts polluted and persists through every program and
        # both stress launches. No program may depend on a predecessor's SPM.
        for bank in machine.spm:
            bank[:] = bytes([0xD3]) * len(bank)
        tail_first = sorted(order, key=lambda pid: (pid[0] != grid_m - 1, pid))
        expected = check_case(
            machine, data, tail_first, limit, "signed-wrap/tail-first"
        )
        assert (
            check_case(machine, data, order[::-1], limit, "signed-wrap/reverse")
            == expected
        )
    m, n = (example.SHAPE[name] for name in ("M", "N"))
    print(
        "SIM CHECK PASS: "
        f"{m * n} outputs match the NumPy INT32 oracle after {total} programs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
