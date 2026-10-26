# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Internal K blocking and dynamic rectangular masks, with an independent oracle."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from Inputs.pipeline import buddy_opt_path, experimental_pipeline  # noqa: E402


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    source = (
        Path(__file__).parent / "Inputs/legacy/k-blocked-64x64x256.ttir"
    ).read_text()
    initial = 2147483630
    source = source.replace(
        "%acc0 = arith.constant dense<0>",
        f"%acc0 = arith.constant dense<{initial}>",
    )
    opt = buddy_opt_path()

    def lower(name, ttir):
        input_path, output, loops = (
            work / (name + suffix)
            for suffix in (".ttir", ".mlir", ".loops.mlir")
        )
        input_path.write_text(ttir)
        result = subprocess.run(
            [
                opt,
                str(input_path),
                *experimental_pipeline(),
                "-o",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        text = output.read_text()
        assert text.count("bosc_ame.tile.mma") == 4, (
            "K=256 must use four 64-wide AME blocks"
        )
        assert (
            "linalg.matmul" not in text and "bosc_ame.lower_to_ame" not in text
        )
        assert "memref.memory_space_cast" not in text
        assert "bufferization.to_tensor" not in text
        assert "bufferization.to_buffer" not in text
        operand_loads = [
            line
            for line in text.splitlines()
            if "bosc_ame.tile.load_a " in line
            or "bosc_ame.tile.load_b " in line
        ]
        assert len(operand_loads) == 8
        assert all(
            "bosc_ame.direct_ddr" in line or "#bosc_ame.spm<" in line
            for line in operand_loads
        ), "AME operands must use a proven direct route or an SPM window"
        subprocess.run(
            [
                opt,
                str(output),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(loops),
            ],
            check=True,
            capture_output=True,
        )
        return Machine(loops, "k_blocked")

    rng = np.random.default_rng(413)
    a = rng.integers(-128, 128, (64, 256), dtype=np.int8)
    b = rng.integers(-128, 128, (64, 256), dtype=np.int8)
    guard = bytes([0xB7]) * 256
    machine = lower("nonzero", source)
    c = bytearray(bytes([0xD3]) * (64 * 64 * 4) + guard)
    buffers = {
        "%arg0": bytearray(a.tobytes() + guard),
        "%arg1": bytearray(b.tobytes() + guard),
        "%arg2": c,
    }
    machine.launch(buffers)
    expected = (a.astype(np.int64) @ b.astype(np.int64).T + initial).astype(
        np.int32
    )
    np.testing.assert_array_equal(
        np.frombuffer(c, dtype=np.int32, count=4096).reshape(64, 64), expected
    )
    assert c[-len(guard) :] == guard
    # CHECK: PASS K blocking: four 64-wide blocks, 4096 exact outputs, nonzero dot initializer and i32 wrap
    print(
        "PASS K blocking: four 64-wide blocks, 4096 exact outputs, nonzero dot initializer and i32 wrap"
    )

    # Dynamic contiguous extent is chunks * 64, so each enabled DMA packet is
    # wholly inside the original mask. B remains physical [N,K], logical [K,N].
    masked = source.replace(
        "%C: !tt.ptr<i32>)", "%C: !tt.ptr<i32>, %chunks: i32)"
    )
    masked = masked.replace(
        "%a = tt.load %pa3 :",
        """%c64scalar = arith.constant 64 : i32
      %valid = arith.muli %chunks, %c64scalar : i32
      %limitA = tt.splat %valid : i32 -> tensor<1x256xi32>
      %maskA0 = arith.cmpi slt, %kn, %limitA : tensor<1x256xi32>
      %maskA = tt.broadcast %maskA0 : tensor<1x256xi1> -> tensor<64x256xi1>
      %otherA = arith.constant dense<3> : tensor<64x256xi8>
      %a = tt.load %pa3, %maskA, %otherA :""",
    )
    masked = masked.replace(
        "%b = tt.load %pb3 :",
        """%limitB = tt.splat %valid : i32 -> tensor<256x1xi32>
      %maskB0 = arith.cmpi slt, %k, %limitB : tensor<256x1xi32>
      %maskB = tt.broadcast %maskB0 : tensor<256x1xi1> -> tensor<256x64xi1>
      %otherB = arith.constant dense<-2> : tensor<256x64xi8>
      %b = tt.load %pb3, %maskB, %otherB :""",
    )
    masked = masked.replace(
        "tt.store %pc3, %dot :",
        """%limitC = arith.constant dense<16> : tensor<1x64xi32>
      %maskC0 = arith.cmpi slt, %n, %limitC : tensor<1x64xi32>
      %maskC = tt.broadcast %maskC0 : tensor<1x64xi1> -> tensor<64x64xi1>
      tt.store %pc3, %dot, %maskC :""",
    )
    dynamic = work / "dynamic-mask.ttir"
    dynamic.write_text(masked)
    rejected = subprocess.run(
        [opt, str(dynamic), *experimental_pipeline()],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "kernel contract V4: static valid shape" in rejected.stderr
    sentinel = np.full((64, 64), 0x3B6A8D0F, dtype=np.int32)
    for chunks in (2, 0, 3, 4):
        # Specialize a supported static rectangular extent while retaining the
        # same original other values and the excluded output columns.
        static_masked = masked.replace(
            "%C: !tt.ptr<i32>, %chunks: i32)", "%C: !tt.ptr<i32>)"
        ).replace(
            "%c64scalar = arith.constant 64 : i32",
            f"%chunks = arith.constant {chunks} : i32\n%c64scalar = arith.constant 64 : i32",
        )
        machine = lower(f"masked-{chunks}", static_masked)
        valid = chunks * 64
        c[:] = sentinel.tobytes() + guard
        first = len(machine.accesses)
        machine.launch(buffers)
        lhs, rhs = a.astype(np.int64), b.astype(np.int64)
        lhs[:, valid:], rhs[:, valid:] = 3, -2
        expected = (lhs @ rhs.T + initial).astype(np.int32)
        actual = np.frombuffer(c, dtype=np.int32, count=4096).reshape(64, 64)
        np.testing.assert_array_equal(actual[:, :16], expected[:, :16])
        np.testing.assert_array_equal(actual[:, 16:], sentinel[:, 16:])
        for name, start, end, kind in machine.accesses[first:]:
            if name in ("%arg0", "%arg1"):
                assert kind == "read" and (
                    valid == 256
                    or (
                        start // 256 == (end - 1) // 256
                        and end - (start // 256) * 256 <= valid
                    )
                )
            else:
                assert (
                    kind == "write"
                    and start // 256 == (end - 1) // 256
                    and end - (start // 256) * 256 == 64
                )
        assert c[-len(guard) :] == guard
        assert (
            buffers["%arg0"] == a.tobytes() + guard
            and buffers["%arg1"] == b.tobytes() + guard
        )
    # CHECK: PASS masks: static 64-byte packets and dynamic-extent rejection, nonzero other, column-major B, excluded DDR bytes and persistent SPM
    print(
        "PASS masks: static 64-byte packets and dynamic-extent rejection, nonzero other, column-major B, excluded DDR bytes and persistent SPM"
    )


if __name__ == "__main__":
    main()
