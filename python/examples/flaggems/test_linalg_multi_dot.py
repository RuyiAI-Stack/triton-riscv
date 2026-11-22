import importlib

import pytest
import torch

from .linalg_multi_dot import linalg_multi_dot, linalg_multi_dot_out


def test_multi_dot_two_and_three_matrices_match_aten():
    torch.manual_seed(0)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    c = torch.randn(4, 5, dtype=torch.float32)
    out = linalg_multi_dot([a, b])
    ref = torch.linalg.multi_dot([a, b])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    out3 = linalg_multi_dot([a, b, c])
    ref3 = torch.linalg.multi_dot([a, b, c])
    assert out3.shape == (2, 5)
    torch.testing.assert_close(out3, ref3, rtol=1e-5, atol=1e-5)


def test_multi_dot_four_matrices_dp_order_matches_aten():
    # Rectangular chain where the optimal parenthesization matters.
    torch.manual_seed(1)
    mats = [
        torch.randn(16, 64, dtype=torch.float32),
        torch.randn(64, 4, dtype=torch.float32),
        torch.randn(4, 128, dtype=torch.float32),
        torch.randn(128, 2, dtype=torch.float32),
    ]
    out = linalg_multi_dot(mats)
    ref = torch.linalg.multi_dot(mats)
    # Three chained fp32 matmuls compound summation-order rounding to ~3.4e-5
    # relative vs aten (measured); 1e-5 would demand bit-identical order.
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_multi_dot_1d_first_and_last_match_aten():
    torch.manual_seed(2)
    v = torch.randn(3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    out = linalg_multi_dot([v, b])
    ref = torch.linalg.multi_dot([v, b])
    assert out.shape == (4,)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    v2 = torch.randn(4, dtype=torch.float32)
    a = torch.randn(2, 4, dtype=torch.float32)
    out2 = linalg_multi_dot([a, v2])
    ref2 = torch.linalg.multi_dot([a, v2])
    assert out2.shape == (2,)
    torch.testing.assert_close(out2, ref2, rtol=1e-5, atol=1e-5)


def test_multi_dot_two_vectors_is_scalar_dot():
    torch.manual_seed(3)
    v1 = torch.randn(5, dtype=torch.float64)
    v2 = torch.randn(5, dtype=torch.float64)
    out = linalg_multi_dot([v1, v2])
    ref = torch.linalg.multi_dot([v1, v2])
    assert out.shape == ()
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)


def test_multi_dot_fp64_arange_exact():
    # Dyadic divisors keep every value/product/sum exactly representable in
    # fp64, so any summation order is bit-exact (divisors like /3 or /5 are
    # not representable and round differently per order).
    mats = [
        torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3) / 2,
        torch.arange(3 * 4, dtype=torch.float64).reshape(3, 4) / 4,
        torch.arange(4 * 5, dtype=torch.float64).reshape(4, 5) / 8,
    ]
    out = linalg_multi_dot(mats)
    ref = torch.linalg.multi_dot(mats)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_multi_dot_fp64_randn():
    # Both sides accumulate in fp64; only the summation order differs.
    # rtol/atol 1e-9 measured against aten fp64 randn chains.
    torch.manual_seed(4)
    mats = [
        torch.randn(6, 6, dtype=torch.float64),
        torch.randn(6, 7, dtype=torch.float64),
        torch.randn(7, 5, dtype=torch.float64),
    ]
    out = linalg_multi_dot(mats)
    torch.testing.assert_close(out, torch.linalg.multi_dot(mats), rtol=1e-9, atol=1e-9)


def test_multi_dot_empty_matrix_matches_aten():
    torch.manual_seed(5)
    a = torch.empty(0, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    out = linalg_multi_dot([a, b])
    ref = torch.linalg.multi_dot([a, b])
    assert out.shape == (0, 4)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_multi_dot_non_contiguous_operand():
    torch.manual_seed(6)
    base = torch.randn(3, 2, dtype=torch.float32)
    a = base.t()  # (2, 3) non-contiguous view
    assert not a.is_contiguous()
    b = torch.randn(3, 4, dtype=torch.float32)
    out = linalg_multi_dot([a, b])
    ref = torch.linalg.multi_dot([a, b])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_multi_dot_errors_match_aten():
    torch.manual_seed(7)
    a = torch.randn(2, 3, dtype=torch.float32)
    c = torch.randn(4, 5, dtype=torch.float32)
    with pytest.raises(
        RuntimeError, match=r"multi_dot\(\): expected at least 2 tensors but got 1"
    ):
        linalg_multi_dot([a])
    with pytest.raises(
        RuntimeError, match=r"multi_dot\(\): tensor 1 must be 2D but got 3D"
    ):
        linalg_multi_dot([a, torch.randn(3, 4, 5), c])
    with pytest.raises(
        RuntimeError, match=r"multi_dot\(\): tensors 0 and 1 .* cannot be multiplied"
    ):
        linalg_multi_dot([a, torch.randn(5, 6)])
    with pytest.raises(
        RuntimeError, match=r"multi_dot\(\): all tensors must have be the same dtype"
    ):
        linalg_multi_dot([a, torch.randn(3, 4, dtype=torch.float64)])


def test_multi_dot_out_writes_buffer():
    torch.manual_seed(8)
    mats = [
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(3, 4, dtype=torch.float32),
        torch.randn(4, 5, dtype=torch.float32),
    ]
    out = torch.empty(2, 5, dtype=torch.float32)
    ret = linalg_multi_dot_out(mats, out=out)
    assert ret is out
    torch.testing.assert_close(out, torch.linalg.multi_dot(mats), rtol=1e-5, atol=1e-5)


def test_multi_dot_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.linalg_multi_dot")
    launches = []
    kernel = mod._multi_dot_mm_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(9)
    mats = [
        torch.randn(4, 8, dtype=torch.float32),
        torch.randn(8, 3, dtype=torch.float32),
    ]
    out = linalg_multi_dot(mats)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.linalg.multi_dot(mats), rtol=1e-5, atol=1e-5)
