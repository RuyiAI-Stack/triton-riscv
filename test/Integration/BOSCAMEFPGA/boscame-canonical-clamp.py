# RUN: %PYTHON %s %t contract-pipeline-opt
"""Canonical rank-one mask clamps retain launch-domain specialization."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import Machine  # noqa: E402


def source(count):
    return """
module {
  tt.func public @add(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %block = arith.constant 128 : i32
    %bound = arith.constant dense<COUNT> : tensor<128xi32>
    %zero = arith.constant dense<0.0> : tensor<128xf32>
    %one = arith.constant 1 : i32
    %pid = tt.get_program_id x : i32
    %row = tt.get_program_id y : i32
    %row_valid = arith.cmpi slt, %row, %one : i32
    %rows_valid = tt.splat %row_valid : i1 -> tensor<128xi1>
    %start = arith.muli %pid, %block : i32
    %splat = tt.splat %start : i32 -> tensor<128xi32>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %index = arith.addi %splat, %range : tensor<128xi32>
    %column_valid = arith.cmpi slt, %index, %bound : tensor<128xi32>
    %valid = arith.andi %rows_valid, %column_valid : tensor<128xi1>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %op = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %ya = tt.addptr %yp, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %oa = tt.addptr %op, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %a = tt.load %xa, %valid, %zero : tensor<128x!tt.ptr<f32>>
    %b = tt.load %ya, %valid, %zero : tensor<128x!tt.ptr<f32>>
    %sum = arith.addf %a, %b : tensor<128xf32>
    tt.store %oa, %sum, %valid : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
""".replace("COUNT", str(count))


def main():
    work, opt = Path(sys.argv[1]), sys.argv[2]
    work.mkdir(parents=True, exist_ok=True)
    for count in (96, 256, 272):
        grid = (count + 127) // 128
        ttir, contract = work / f"{count}.ttir", work / f"{count}.mlir"
        ttir.write_text(source(count))
        subprocess.run(
            [
                opt,
                str(ttir),
                f"--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias launch-grid=add:{grid},1,1",
                "-o",
                str(contract),
            ],
            check=True,
            capture_output=True,
        )
        text = contract.read_text()
        assert "memref<?xf32" not in text and "arith.minsi" not in text
        assert text.count("scf.if") == int(count == 272)
        assert "contract.compute_graph = {complete = true" in text
        if count != 256:
            assert f"valid_shape = array<i64: {count % 128}>" in text
        emitted = work / f"{count}.boscame.mlir"
        subprocess.run(
            [
                opt,
                str(contract),
                "--bosc-ame-plan",
                "--bosc-ame-plan-verify",
                "--bosc-ame-emit",
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(emitted),
            ],
            check=True,
            capture_output=True,
        )
        machine = Machine(emitted, "add")
        x = (np.arange(count) * 0.25).astype("<f4")
        y = (np.arange(count) * -0.125).astype("<f4")
        guard = bytes([0xD7]) * 64
        buffers = {
            "%arg0": bytearray(x.tobytes() + guard),
            "%arg1": bytearray(y.tobytes() + guard),
            "%arg2": bytearray(bytes(count * 4) + guard),
        }
        for pid in reversed(range(grid)):
            machine.launch(
                buffers,
                {
                    "%arg3": grid,
                    "%arg4": 1,
                    "%arg5": 1,
                    "%arg6": pid,
                    "%arg7": 0,
                    "%arg8": 0,
                },
            )
        actual = np.frombuffer(buffers["%arg2"], dtype="<f4", count=count)
        np.testing.assert_array_equal(actual, x + y)
        assert buffers["%arg0"] == x.tobytes() + guard
        assert buffers["%arg1"] == y.tobytes() + guard
        assert buffers["%arg2"][-64:] == guard
        assert all(0 <= a < b <= count * 4 for _, a, b, _ in machine.accesses)
    # Without a launch domain the mask remains dynamic; do not invent a
    # full-tile proof merely because its canonical outer min is absent.
    result = subprocess.run(
        [
            opt,
            str(ttir),
            "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode and "kernel contract V4:" in result.stderr
    # A wider Y domain makes row < 1 genuinely conditional. A known grid is
    # not permission to assume that predicate true for every program.
    result = subprocess.run(
        [
            opt,
            str(ttir),
            "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias launch-grid=add:3,2,1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode and "kernel contract V4:" in result.stderr


if __name__ == "__main__":
    main()
