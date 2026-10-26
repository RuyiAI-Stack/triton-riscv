# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Real exported TTIR, persistent SPM, guards, and a select-zero oracle."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from Inputs.pipeline import buddy_opt_path, experimental_pipeline  # noqa: E402


def lower(source, output, boscame):
    opt = buddy_opt_path()
    command = [
        opt,
        str(source),
        *(
            experimental_pipeline()
            if boscame
            else ["--triton-to-linalg-experimental"]
        ),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    loops = str(output) + ".loops.mlir"
    command = [opt, str(output)]
    if not boscame:
        command += [
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries=true",
            "--promote-buffers-to-stack",
        ]
    command += [
        "--convert-linalg-to-loops",
        "--lower-affine",
        "--canonicalize",
        "-o",
        loops,
    ]
    subprocess.run(command, check=True, capture_output=True)
    return loops


def main():
    prefix = Path(sys.argv[1])
    source = Path(__file__).parent / "Inputs/layernorm.ttir"
    candidate = lower(source, str(prefix) + ".boscame.mlir", True)
    baseline = lower(source, str(prefix) + ".default.mlir", False)
    candidate_text = Path(str(prefix) + ".boscame.mlir").read_text()
    assert candidate_text.count("vector.reduction <add>") == 2
    assert "vector.extract " not in candidate_text
    assert "vector<1xf32>" not in candidate_text
    # Scalar reduction seeds/results are ordinary scratch. All tensor data
    # stays in SPM; the two f32 slots add only eight bytes of stack payload.
    assert (
        len(re.findall(r"memref\.alloca[^\n]*: memref<f32>", candidate_text))
        == 2
    )
    assert "memref.copy" not in candidate_text
    assert all(
        "memref<f32>" in line and "#bosc_ame.spm" not in line
        for line in candidate_text.splitlines()
        if re.search(r"memref\.(load|store)\b", line)
    )
    assert (
        len(
            re.findall(
                r"vector\.reduction <add>[^\n]*vector<16xf32> into f32",
                candidate_text,
            )
        )
        == 2
    )
    assert (
        len(
            re.findall(
                r"scf\.for[^\n]*iter_args\([^\n]*-> \(f32\)", candidate_text
            )
        )
        == 2
    )
    for operation in ("subf", "mulf", "addf"):
        assert re.search(
            rf"arith\.{operation}[^\n]*: vector<\d+xf32>", candidate_text
        ), "LayerNorm computation did not consume SPM vectors"
    predicate_transfers = 0
    for line in candidate_text.splitlines():
        if "vector.load " in line or "vector.store " in line:
            if re.search(r"vector<\d+xi1>", line):
                predicate_transfers += 1
                assert "#bosc_ame.spm<" not in line, (
                    "predicate vectors must not become SPM data transfers"
                )
            else:
                assert "#bosc_ame.spm<" in line, (
                    "LayerNorm data vectors must use SPM buffers"
                )
    assert predicate_transfers == 0, (
        "LayerNorm static prefix select should not materialize predicate vectors"
    )
    assert not re.search(r"bosc_ame\.dma\.start[^\n]*i1", candidate_text), (
        "predicate-only i1 buffers must not become DMA payloads"
    )
    assert not re.search(
        r"bosc_ame\.dma\.start[^\n]*memref<\d+xi32", candidate_text
    ), "static index and bound auxiliaries must not become DMA payloads"
    assert "memref.memory_space_cast" not in candidate_text
    rng = np.random.default_rng(71)
    rows, width = 197, 768
    x = rng.normal(0.8, 1.5, (rows, width)).astype(np.float32)
    weight = rng.uniform(-1, 2, width).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, width).astype(np.float32)
    guard = bytes([0xC7]) * 1024
    buffers = {
        "%arg0": bytearray(x.tobytes() + guard),
        "%arg1": bytearray(weight.tobytes() + guard),
        "%arg2": bytearray(bias.tobytes() + guard),
        "%arg3": bytearray(bytes([0xDA]) * x.nbytes + guard),
    }
    reference_buffers = {
        name: bytearray(data) for name, data in buffers.items()
    }
    snapshots = {
        name: bytes(data) for name, data in buffers.items() if name != "%arg3"
    }
    machine = Machine(candidate, "triton_layernorm_197x768")
    reference = Machine(baseline, "triton_layernorm_197x768")
    # Reversed/interleaved programs reuse poisoned SPM banks and include the last row.
    order = list(range(0, rows, 2)) + list(range(1, rows, 2))
    if os.environ.get("BOSCAME_NUMERIC_QUICK"):
        order = [196, 0, 1]
    max_error = 0.0
    for row in order:
        scalars = {"%arg4": rows, "%arg5": 1, "%arg6": 1, "%arg7": row}
        machine.launch(buffers, scalars)
        reference.launch(reference_buffers, scalars)
        start, end = row * width * 4, (row + 1) * width * 4
        actual = np.frombuffer(buffers["%arg3"][start:end], dtype=np.float32)
        reference_actual = np.frombuffer(
            reference_buffers["%arg3"][start:end], dtype=np.float32
        )
        np.testing.assert_allclose(
            actual,
            reference_actual,
            rtol=2.0e-5,
            atol=2.0e-5,
            err_msg="default experimental path differs beyond vector-reduction tolerance",
        )
        # The second reduction keeps 1024 lanes after selecting padded lanes to zero.
        padded = np.pad(x[row].astype(np.float64), (0, 256))
        mean = padded.sum() / width
        centered = np.where(np.arange(1024) < width, padded - mean, 0.0)
        variance = np.square(centered).sum() / width
        expected = (
            centered[:width] / np.sqrt(variance + np.float32(1.0e-12))
        ) * weight + bias
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        # An independent f32 oracle spells out each TTIR operation and the
        # sequential reduction selected by the unchanged Linalg loop path.
        # Vector reduction may round differently on the target. The f64
        # formula above separately measures accumulated rounding, including
        # cancellation in the affine epilogue near zero.
        padded32 = np.pad(x[row], (0, 256))
        mean32 = np.float32(
            np.cumsum(padded32, dtype=np.float32)[-1] / np.float32(width)
        )
        centered32 = np.subtract(padded32, mean32, dtype=np.float32)
        centered32[width:] = np.float32(0)
        squares32 = np.multiply(centered32, centered32, dtype=np.float32)
        var32 = np.float32(
            np.cumsum(squares32, dtype=np.float32)[-1] / np.float32(width)
        )
        shifted32 = np.float32(var32 + np.float32(1.0e-12))
        # math.rsqrt lowers to sqrtf followed by a separate f32 reciprocal.
        root32 = np.sqrt(shifted32, dtype=np.float32)
        inverse32 = np.divide(np.float32(1.0), root32, dtype=np.float32)
        oracle32 = np.add(
            np.multiply(
                np.multiply(centered32[:width], inverse32, dtype=np.float32),
                weight,
                dtype=np.float32,
            ),
            bias,
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, oracle32, rtol=2.0e-5, atol=2.0e-5)
        accesses = machine.accesses[-4:]
        assert accesses == [
            ("%arg0", start, end, "read"),
            ("%arg1", 0, width * 4, "read"),
            ("%arg2", 0, width * 4, "read"),
            ("%arg3", start, end, "write"),
        ], accesses
    for name, snapshot in snapshots.items():
        assert bytes(buffers[name]) == snapshot, "input/guard modified"
    assert bytes(buffers["%arg3"][-len(guard) :]) == guard
    # CHECK: PASS LayerNorm: 197 programs, three 3072 B loads and one 3072 B store per program
    print(
        f"PASS LayerNorm: {len(order)} programs, three 3072 B loads and one 3072 B store per program"
    )
    # CHECK: PASS numerics: rtol=2e-5, atol=2e-5 against default experimental buffer IR and independent masked-lane TTIR oracle
    print(
        f"PASS numerics: rtol=2e-5, atol=2e-5 against default experimental buffer IR and independent masked-lane TTIR oracle (max absolute error vs f64 {max_error:.3g})"
    )
    # CHECK: PASS guards: last row, Weight/Bias tails and persistent poisoned SPM
    print(
        "PASS guards: last row, Weight/Bias tails and persistent poisoned SPM"
    )
    # CHECK: PASS SPM: direct vector arithmetic and two ordered vector reductions
    print(
        "PASS SPM: direct vector arithmetic and two ordered vector reductions"
    )


if __name__ == "__main__":
    main()
