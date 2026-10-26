# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Full example grid: signed i8, physical [N,K] B, nonzero C and i32 wrap."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from Inputs.pipeline import (  # noqa: E402
    buddy_opt_path,
    experimental_pipeline,
)


def main():
    source = Path(__file__).parent / "Inputs/matmul-i8.ttir"
    output = sys.argv[1] + ".boscame.mlir"
    loops = sys.argv[1] + ".loops.mlir"
    opt = buddy_opt_path()
    dynamic = subprocess.run(
        [opt, str(source), *experimental_pipeline()],
        capture_output=True,
        text=True,
    )
    assert dynamic.returncode != 0
    assert "kernel contract V4: static valid shape" in dynamic.stderr
    subprocess.run(
        [
            opt,
            str(source),
            *experimental_pipeline("triton_matmul_i8_128x1024x128:8,64,1"),
            "-o",
            output,
        ],
        check=True,
        capture_output=True,
    )
    text = Path(output).read_text()
    assert "bosc_ame.tile.mma" in text and "linalg.matmul" not in text
    assert "bosc_ame.lower_to_ame" not in text
    # This legacy source loads C after the K loop. Keep that ordering: A/B
    # forward to DDR, while the loop result and C snapshot use private SPM.
    assert text.count("bosc_ame.tile.load_acc ") == 1
    assert text.count("bosc_ame.tile.load_a ") == 1
    assert text.count("bosc_ame.tile.load_b ") == 1
    assert text.count("bosc_ame.tile.mma ") == 1
    assert text.count("bosc_ame.tile.store_acc ") == 1
    assert text.index("scf.for ") < text.index("bosc_ame.tile.load_acc ")
    assert text.index("scf.for ") < text.index("bosc_ame.tile.store_acc ")
    assert not any(
        "iter_args" in line and "vector<" in line
        for line in text.splitlines()
        if "scf.for " in line
    )
    assert re.search(r"arith.addi[^\n]*vector<16xi32>", text)
    for line in text.splitlines():
        if "bosc_ame.tile.load_a " in line or "bosc_ame.tile.load_b " in line:
            assert "bosc_ame.direct_ddr" in line, line
        elif (
            "bosc_ame.tile.load_acc " in line
            or "bosc_ame.tile.store_acc " in line
        ):
            assert "#bosc_ame.spm<" in line, line
    assert text.count("bosc_ame.spm.view") == 2
    assert text.count("bosc_ame.dma.start") == 32
    assert "memref.memory_space_cast" not in text
    assert "bufferization.to_tensor" not in text
    assert "bufferization.to_buffer" not in text
    subprocess.run(
        [
            opt,
            output,
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
            "-o",
            loops,
        ],
        check=True,
        capture_output=True,
    )
    rng = np.random.default_rng(923)
    m, n, k = 128, 1024, 128
    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (n, k), dtype=np.int8)
    # Alternating extrema and nonsymmetric inputs distinguish signedness/layout.
    a[0, ::2] = -128
    b[0, 1::2] = 127
    initial = rng.integers(-(1 << 31), 1 << 31, (m, n), dtype=np.int32)
    initial[0, :16] = np.iinfo(np.int32).max
    guard = bytes([0xB9]) * 1024
    buffers = {
        "%arg0": bytearray(a.tobytes() + guard),
        "%arg1": bytearray(b.tobytes() + guard),
        "%arg2": bytearray(initial.tobytes() + guard),
    }
    snapshots = {
        name: bytes(data) for name, data in buffers.items() if name != "%arg2"
    }
    machine = Machine(loops, "triton_matmul_i8_128x1024x128")
    # The independent oracle uses wide exact accumulation and wraps only at
    # the final i32 conversion, never any generated staging or tile indices.
    expected = (
        a.astype(np.int64) @ b.astype(np.int64).T + initial.astype(np.int64)
    ).astype(np.int32)
    for i in range(m // 16):
        for j in range(n // 16):
            machine.launch(
                buffers,
                {
                    "%arg3": m // 16,
                    "%arg4": n // 16,
                    "%arg5": 1,
                    "%arg6": i,
                    "%arg7": j,
                },
            )
    actual = np.frombuffer(
        buffers["%arg2"], dtype=np.int32, count=m * n
    ).reshape(m, n)
    np.testing.assert_array_equal(actual, expected)
    assert bytes(buffers["%arg2"][-len(guard) :]) == guard
    for name, snapshot in snapshots.items():
        assert bytes(buffers[name]) == snapshot
    # Only user ranges are logged; ordinary packing buffers have separate storage.
    limits = {"%arg0": a.nbytes, "%arg1": b.nbytes, "%arg2": initial.nbytes}
    assert all(
        0 <= start < end <= limits[name]
        for name, start, end, _ in machine.accesses
    )
    assert all(
        name == "%arg2"
        for name, _, _, kind in machine.accesses
        if kind == "write"
    )
    c_accesses = [
        (name, kind) for name, _, _, kind in machine.accesses if name == "%arg2"
    ]
    # C is read and written once per program, in 16 contiguous row packets.
    expected_c_rows = (m // 16) * (n // 16) * 16
    assert sum(kind == "read" for _, kind in c_accesses) == expected_c_rows
    assert sum(kind == "write" for _, kind in c_accesses) == expected_c_rows
    assert len(machine.dma_log) == expected_c_rows * 2
    # CHECK: PASS GEMM: 512 programs, 131072 exact i32 outputs (signed i8, nonzero C, wrap, physical B)
    print(
        "PASS GEMM: 512 programs, 131072 exact i32 outputs (signed i8, nonzero C, wrap, physical B)"
    )
    # CHECK: PASS guards: no user DDR overread/overwrite, persistent SPM, two outer K blocks
    print(
        "PASS guards: no user DDR overread/overwrite, persistent SPM, two outer K blocks"
    )


if __name__ == "__main__":
    main()
