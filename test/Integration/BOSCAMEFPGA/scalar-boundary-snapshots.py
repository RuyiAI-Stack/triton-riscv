# RUN: %PYTHON %s %t contract-pipeline-opt
"""Scalar scale accesses reuse neutral singleton boundary snapshots."""

import os
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import Machine  # noqa: E402

work, tool = Path(sys.argv[1]), sys.argv[2]
work.mkdir(parents=True, exist_ok=True)
source = work / "scalar.ttir"
contract = work / "scalar.contract.mlir"
emitted = work / "scalar.boscame.mlir"
source.write_text(
    """
module {
  tt.func public @scalar(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %source = tt.addptr %input, %pid : !tt.ptr<f32>, i32
    %value = tt.load %source : !tt.ptr<f32>
    %destination = tt.addptr %output, %pid : !tt.ptr<f32>, i32
    tt.store %destination, %value : !tt.ptr<f32>
    tt.return
  }
}
"""
)
subprocess.run(
    [
        tool,
        str(source),
        "--triton-to-linalg-experimental=contract=true "
        "caller-contract=args_aligned64_nonalias launch-grid=scalar:3,1,1",
        "--kernel-contract-verify",
        "-o",
        str(contract),
    ],
    check=True,
    capture_output=True,
)
text = contract.read_text()
assert "complete = true" in text
assert "logical_shape = array<i64: 1>" in text
assert "memref<1xf32>" in text
assert "tts." not in text and "tt." not in text
subprocess.run(
    [
        tool,
        str(contract),
        "--bosc-ame-plan",
        "--bosc-ame-plan-verify",
        "--bosc-ame-emit",
        "--canonicalize",
        "-o",
        str(emitted),
    ],
    check=True,
    capture_output=True,
)
guard = bytes([0xD7]) * 64
values = struct.pack("<3f", -0.0, -2.25, 127.0)
buffers = {
    "%arg0": bytearray(values + guard),
    "%arg1": bytearray(bytes([0xA5]) * len(values) + guard),
}
machine = Machine(emitted, "scalar")
for bank in machine.spm:
    bank[:] = bytes([0xB9]) * len(bank)
for pid in (2, 0, 1):
    machine.launch(
        buffers,
        {
            "%arg2": 3,
            "%arg3": 1,
            "%arg4": 1,
            "%arg5": pid,
            "%arg6": 0,
            "%arg7": 0,
        },
    )
assert buffers["%arg0"] == values + guard
assert buffers["%arg1"] == values + guard
assert all(end - begin == 4 for _, begin, end, _ in machine.accesses)
