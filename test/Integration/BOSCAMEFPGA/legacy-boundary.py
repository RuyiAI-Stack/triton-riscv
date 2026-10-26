# RUN: %PYTHON %s %t
"""Retired scratch-ABI inputs through the standard two-process boundary.

Dense and pitched inputs execute emitted IR with strict byte-range checks.
K-blocking, initializer/wrap and static/nonzero-other mask numerics are also
covered by boscame-k-blocked.py. Dynamic old scratch masks remain explicit v1
rejections; no scratch argument or legacy frontend pass is introduced.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ["BUDDY_ROOT"])
sys.path.insert(0, str(REPO / "examples/FPGA-BOSCAME/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from Inputs.pipeline import buddy_opt_path, experimental_pipeline  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
opt = buddy_opt_path()
fixtures = Path(__file__).parent / "Inputs/legacy"
expected = {
    "diag-coupled-mask": "unstructured/scalar memory access is unsupported",
    "diag-dynamic-other": "unstructured/scalar memory access is unsupported",
    "diag-mask-coefficient": "unstructured/scalar memory access is unsupported",
    # The new planner forwards these operands, avoiding the old scratch budget.
    "diag-spm-capacity": None,
    "diag-too-wide": "AME plan requires frontend tiling/padding",
    "diag-transposed-b": "AME tile addresses or parent pitch are not legal",
    "k-blocked-64x64x256": None,
    "masked-a-stride128": "unstructured/scalar memory access is unsupported",
    "padded-k-initializer": "unstructured/scalar memory access is unsupported",
    "reference-matmul-i8-196x768x768": None,
    "single-dot-64x64x64": None,
    "unaligned-a": None,
}
assert {path.stem for path in fixtures.glob("*.ttir")} == expected.keys()
for name, diagnostic in expected.items():
    grid = (
        "triton_matmul_i8_196x768x768:7,6,1"
        if name.startswith("reference-")
        else None
    )
    output = work / (name + ".mlir")
    result = subprocess.run(
        [
            opt,
            str(fixtures / (name + ".ttir")),
            *experimental_pipeline(grid),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode >= 0 and "Stack dump" not in result.stderr, (
        result.stderr
    )
    if diagnostic:
        assert result.returncode != 0 and diagnostic in result.stderr, (
            name,
            result.stderr,
        )
        continue
    assert result.returncode == 0, (name, result.stderr)
    text = output.read_text()
    assert "tt." not in text and "tptr." not in text and "plan." not in text
    if name not in ("single-dot-64x64x64", "unaligned-a"):
        continue
    loops = work / (name + ".loops.mlir")
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
    )
    # Discover the exact existing symbol; no kernel text is rewritten.
    function = re.search(r"func.func @([\w]+)", text)[1]
    machine = Machine(loops, function)
    rng = np.random.default_rng(349)
    pitch = 100 if name == "unaligned-a" else 64
    guard = bytes([0xD7]) * 128
    for _ in range(2):
        a = rng.integers(-128, 128, (64, pitch), dtype=np.int8)
        b = rng.integers(-128, 128, (64, pitch), dtype=np.int8)
        c = bytearray(bytes([0xA5]) * (64 * 64 * 4) + guard)
        buffers = {
            "%arg0": bytearray(a.tobytes() + guard),
            "%arg1": bytearray(b.tobytes() + guard),
            "%arg2": c,
        }
        first = len(machine.accesses)
        machine.launch(buffers)
        reference = (
            a[:, :64].astype(np.int64) @ b[:, :64].astype(np.int64).T
        ).astype(np.int32)
        np.testing.assert_array_equal(
            np.frombuffer(c, dtype=np.int32, count=4096).reshape(64, 64),
            reference,
        )
        assert c[-128:] == guard
        assert buffers["%arg0"] == a.tobytes() + guard
        assert buffers["%arg1"] == b.tobytes() + guard
        for argument, start, end, kind in machine.accesses[first:]:
            if argument == "%arg2":
                assert kind == "write" and 0 <= start < end <= 64 * 64 * 4
            else:
                assert kind == "read" and 0 <= start < end <= 64 * pitch
                assert start // pitch == (end - 1) // pitch
                assert (end - 1) % pitch < 64
print(
    "PASS legacy boundary: 12 original inputs, dense/pitched exact numerics, guards and persistent SPM"
)
