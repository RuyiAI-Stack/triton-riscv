import importlib

import pytest
import torch

from ._compute_linear_combination import (
    _compute_linear_combination,
    _compute_linear_combination_out,
)

# torch reference: aten::_compute_linear_combination (.out), CPU-callable.
torch_clc = torch._compute_linear_combination

# (coefficients shape [m, n], input tail shape) - the contraction length is n.
CASES = [
    ((3, 4), ()),  # 1-d input
    ((3, 4), (8,)),  # 2-d
    ((1, 4), (8,)),  # single coefficient row
    ((3, 1), (8,)),  # single-element contraction
    ((5, 4), (2, 3)),  # 3-d input, NTAIL == 2
    ((3, 4), (2, 3, 5)),  # NTAIL == 3
    ((3, 4), (2, 3, 5, 2)),  # NTAIL == 4
    ((2, 64), (16,)),  # K > BLOCK_K, loop over contraction blocks
    ((3, 4), (300,)),  # P > BLOCK_P, grid split along the tail
    ((0, 4), (8,)),  # empty coefficient table (m == 0)
]


def make_input(n, tail, dtype=torch.float32):
    torch.manual_seed(0)
    m = 3
    coefficients = torch.randn(m, n, dtype=dtype)
    input = torch.randn((n,) + tail, dtype=dtype)
    return input, coefficients


@pytest.mark.parametrize(
    "coeff_shape, tail",
    CASES,
    ids=[
        "1d",
        "2d",
        "single-row",
        "single-k",
        "3d",
        "4d",
        "5d",
        "wide-k",
        "wide-p",
        "empty-m",
    ],
)
def test_compute_linear_combination_matches_aten(coeff_shape, tail):
    m, n = coeff_shape
    torch.manual_seed(0)
    input = torch.randn((n,) + tail)
    coefficients = torch.randn(m, n)

    ref = torch_clc(input, coefficients)
    res = _compute_linear_combination(input, coefficients)

    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    torch.testing.assert_close(res, ref, rtol=1e-5, atol=1e-5)


def test_compute_linear_combination_einsum_semantics():
    # output[i, ...] = sum_j coefficients[i, j] * input[j, ...]
    torch.manual_seed(1)
    input = torch.randn(4, 6, 5)
    coefficients = torch.randn(7, 4)

    res = _compute_linear_combination(input, coefficients)
    ref = torch.einsum("ij,jkl->ikl", coefficients, input)
    torch.testing.assert_close(res, ref, rtol=1e-5, atol=1e-5)


def test_compute_linear_combination_single_element():
    torch.manual_seed(0)
    input = torch.randn(1)
    coefficients = torch.randn(2, 1)
    res = _compute_linear_combination(input, coefficients)
    ref = torch_clc(input, coefficients)
    assert res.shape == (2,)
    torch.testing.assert_close(res, ref, rtol=1e-6, atol=1e-6)


def test_compute_linear_combination_zero_coefficients():
    input = torch.randn(4, 8)
    coefficients = torch.zeros(3, 4)
    res = _compute_linear_combination(input, coefficients)
    assert torch.count_nonzero(res) == 0


def test_compute_linear_combination_non_contiguous_input():
    # Arbitrary input strides are honored (the tail slots decode them).
    torch.manual_seed(2)
    input = torch.randn(8, 4, 6).transpose(0, 1)  # (4, 8, 6) strided
    coefficients = torch.randn(3, 4)

    res = _compute_linear_combination(input, coefficients)
    ref = torch.einsum("ij,jkl->ikl", coefficients, input)
    torch.testing.assert_close(res, ref, rtol=1e-5, atol=1e-5)


def test_compute_linear_combination_non_contiguous_coefficients():
    # aten's CPU kernel silently misreads strided coefficients (probed);
    # the port walks COEFF_SM/SK, so compare against the exact einsum.
    torch.manual_seed(3)
    input = torch.randn(6, 8)
    coefficients = torch.randn(6, 4).t()  # (4, 6) view, non-contiguous

    res = _compute_linear_combination(input, coefficients)
    ref = coefficients @ input
    torch.testing.assert_close(res, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.int64, 0, 0),
    ],
)
def test_compute_linear_combination_dtypes(dtype, rtol, atol):
    torch.manual_seed(4)
    if dtype == torch.int64:
        input = torch.arange(40).reshape(4, 10)
        coefficients = torch.arange(8).reshape(2, 4)
        ref = torch_clc(input, coefficients)
    else:
        input = torch.randn(4, 10, dtype=dtype)
        coefficients = torch.randn(2, 4, dtype=dtype)
        ref = torch_clc(input, coefficients)

    res = _compute_linear_combination(input, coefficients)
    assert res.dtype == dtype
    torch.testing.assert_close(res, ref, rtol=rtol, atol=atol)


def test_compute_linear_combination_out_accumulates():
    # aten's .out variant ADDS into the existing buffer (probed); the port
    # mirrors it through ACCUM=True.
    torch.manual_seed(0)
    input = torch.randn(4, 8)
    coefficients = torch.randn(3, 4)

    ref_buf = torch.full((3, 8), 5.0)
    torch_clc(input, coefficients, out=ref_buf)

    buf = torch.full((3, 8), 5.0)
    res = _compute_linear_combination(input, coefficients, out=buf)

    assert res is buf
    torch.testing.assert_close(buf, ref_buf, rtol=1e-5, atol=1e-5)


def test_compute_linear_combination_out_non_contiguous():
    # A strided out view is staged through a temporary and written back.
    torch.manual_seed(0)
    input = torch.randn(4, 8)
    coefficients = torch.randn(3, 4)

    ref_pad = torch.zeros(3, 16)
    torch_clc(input, coefficients, out=ref_pad[:, :8])

    pad = torch.zeros(3, 16)
    view = pad[:, :8]
    res = _compute_linear_combination(input, coefficients, out=view)

    assert res is view
    torch.testing.assert_close(view, ref_pad[:, :8], rtol=1e-5, atol=1e-5)
    assert torch.equal(pad[:, 8:], torch.zeros(3, 8))


def test_compute_linear_combination_out_variant_alias():
    torch.manual_seed(0)
    input = torch.randn(4, 8)
    coefficients = torch.randn(3, 4)

    direct = _compute_linear_combination(input, coefficients)
    buf = torch.zeros(3, 8)
    res = _compute_linear_combination_out(input, coefficients, out=buf)

    assert res is buf
    torch.testing.assert_close(buf, direct, rtol=1e-5, atol=1e-5)

    # aten's .out accumulates into the caller's buffer (probed on torch 2.15):
    # out == old_out + coefficients @ input.
    accumulated = torch.full((3, 8), 7.0)
    _compute_linear_combination_out(input, coefficients, out=accumulated)
    torch.testing.assert_close(accumulated, direct + 7.0, rtol=1e-5, atol=1e-5)


def test_compute_linear_combination_out_wrong_shape():
    torch.manual_seed(0)
    input = torch.randn(4, 8)
    coefficients = torch.randn(3, 4)
    with pytest.raises(AssertionError):
        _compute_linear_combination(input, coefficients, out=torch.empty(4, 8))


def test_compute_linear_combination_empty_input():
    # aten raises RuntimeError("Empty tensor not supported"); the port
    # mirrors the message (the upstream module asserts instead).
    input = torch.randn(0, 8)
    coefficients = torch.randn(3, 0)
    with pytest.raises(RuntimeError, match="Empty tensor not supported"):
        torch_clc(input, coefficients)
    with pytest.raises(RuntimeError, match="Empty tensor not supported"):
        _compute_linear_combination(input, coefficients)


def test_compute_linear_combination_empty_tail():
    input = torch.randn(4, 0)
    coefficients = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="Empty tensor not supported"):
        _compute_linear_combination(input, coefficients)


def test_compute_linear_combination_mismatched_contraction():
    # aten's CPU kernel tolerates a mismatch (probed: silently uses
    # input.size(0)); the port keeps the upstream shape assert.
    torch.manual_seed(0)
    input = torch.randn(4, 8)
    coefficients = torch.randn(3, 5)
    with pytest.raises(AssertionError):
        _compute_linear_combination(input, coefficients)


def test_compute_linear_combination_rejects_non_2d_coefficients():
    torch.manual_seed(0)
    input = torch.randn(4, 8)
    with pytest.raises(AssertionError):
        _compute_linear_combination(input, torch.randn(4))


def test_compute_linear_combination_result_is_fresh_buffer():
    torch.manual_seed(0)
    input = torch.randn(4, 8)
    coefficients = torch.randn(3, 4)
    res = _compute_linear_combination(input, coefficients)
    assert res.is_contiguous()
    assert res.data_ptr() != input.data_ptr()
    assert res.data_ptr() != coefficients.data_ptr()


def test_compute_linear_combination_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._compute_linear_combination")

    launches = []
    real_run = mod._compute_linear_combination_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._compute_linear_combination_kernel, "run", spy_run)
    torch.manual_seed(0)
    input = torch.randn(4, 300)
    coefficients = torch.randn(5, 4)
    res = mod._compute_linear_combination(input, coefficients)

    assert len(launches) >= 1
    assert launches[0][0] == 1  # cdiv(m=5, BLOCK_M)
    assert launches[0][1] == 2  # cdiv(P=300, BLOCK_P=256)
    torch.testing.assert_close(
        res, torch_clc(input, coefficients), rtol=1e-5, atol=1e-5
    )
