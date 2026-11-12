"""Python lit test 'vit-pv-oracle.py' covering ViT attention PV CPU oracle.

Inputs: triton-shared-opt (and sometimes the Triton Python plugin) passed via
lit substitutions.

Outputs: assertions / FileCheck on exported contract IR or numeric results;
non-zero exit on mismatch.

When used: BOSCAME FPGA integration suite. Keep RUN lines and inner docstrings
that describe the scenario.
"""

# RUN: %PYTHON %s
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/BOSCAMEFPGA/vit-base-patch16-224/tools"))
from oracles import (  # noqa: E402
    _matmul_blocked_sequential,
    _matmul_sequential,
    attention_fp32_reference,
    quantized_attention_reference,
    quantize_rows,
)


def test_quantized_host_baseline():
    rng = np.random.default_rng(71)
    for family, k, n, factor in (
        ("attention_qk", 64, 197, 0.125),
        ("attention_pv", 197, 64, 1.0),
    ):
        case = {"family": family, "shape": [2, 3, n, k]}
        arrays = {
            "A": rng.normal(size=(2, 3, k)).astype(np.float32),
            "B": rng.normal(size=(2, k, n)).astype(np.float32),
            "C": np.full((2, 3, n), 12345, np.float32),
        }
        arrays["A"][0, 0] = 0
        expected = quantized_attention_reference(case, arrays)["C"]
        independent = np.empty_like(expected)
        for head in range(2):
            aq, rs = quantize_rows(arrays["A"][head])
            bq, cs = quantize_rows(arrays["B"][head].T)
            for row in range(3):
                for col in range(n):
                    acc = sum(int(aq[row, x]) * int(bq[col, x]) for x in range(k))
                    independent[head, row, col] = np.float32(
                        np.float32(np.float32(np.float32(acc) * rs[row]) * cs[col])
                        * np.float32(factor)
                    )
        assert np.array_equal(expected.view(np.uint32), independent.view(np.uint32))
        assert np.all(expected[0, 0] == 0)
        assert np.array_equal(
            expected,
            quantized_attention_reference(
                case, {**arrays, "C": np.full_like(expected, -777)}
            )["C"],
        )
        baseline = attention_fp32_reference(case, arrays)
        assert np.all(np.isfinite(baseline)) and np.any(expected != baseline)
    values = np.array([[0, 0, 0, 0, 0], [-127, -0.5, 0, 0.5, 127]], np.float32)
    quantized, scales = quantize_rows(values)
    assert np.array_equal(scales, [1, 1]) and np.array_equal(
        quantized[1], [-127, -1, 0, 1, 127]
    )
    boundary = np.array(
        [
            [
                -127,
                np.nextafter(np.float32(0.5), np.float32(0)),
                0.5,
                np.nextafter(np.float32(-0.5), np.float32(0)),
                127,
            ]
        ],
        np.float32,
    )
    assert np.array_equal(quantize_rows(boundary)[0], [[-127, 0, 1, 0, 127]])


def test_block_order():
    for k in (197, 192):
        a = np.zeros((1, k), np.float32)
        b = np.ones((k, 1), np.float32)
        a[0, 0], a[0, 64], a[0, 65] = 2**25, -(2**25), 1
        assert _matmul_blocked_sequential(a, b, 64).item() == 0
        assert _matmul_sequential(a, b).item() == 1
    a = np.array([[1, 2, -4, 8, 0.5]], np.float32)
    b = np.ones((5, 1), np.float32)
    assert _matmul_blocked_sequential(a, b, 64).item() == 7.5
    try:
        _matmul_blocked_sequential(a, b, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive block accepted")


def test_no_fma():
    a = np.array([[-1, 1 + 2**-23]], np.float32)
    b = np.array([[1], [1 - 2**-23]], np.float32)
    assert _matmul_blocked_sequential(a, b, 64).item() == 0
    fused_control = np.float32(float(a[0, 1]) * float(b[1, 0]) - 1)
    assert fused_control == np.float32(-(2**-46))


def test_padding_and_special_values():
    a = np.array([[np.inf, 0, -0.0, 0, 0]], np.float32)
    b = np.ones((5, 1), np.float32)
    original = _matmul_sequential
    shapes = []
    import oracles

    def observe(left, right):
        shapes.append((left.shape, right.shape))
        assert np.all(left[..., 5:] == 0)
        assert not np.any(np.signbit(left[..., 5:]))
        assert np.all(right[..., 5:, :] == 0)
        assert not np.any(np.signbit(right[..., 5:, :]))
        return original(left, right)

    oracles._matmul_sequential = observe
    try:
        assert np.isposinf(_matmul_blocked_sequential(a, b, 64).item())
        a[0, 0] = np.nan
        assert np.isnan(_matmul_blocked_sequential(a, b, 64).item())
    finally:
        oracles._matmul_sequential = original
    assert shapes == [((1, 64), (64, 1))] * 2


test_quantized_host_baseline()
test_block_order()
test_no_fma()
test_padding_and_special_values()
