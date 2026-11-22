import importlib

import pytest
import torch

from .tensordot import tensordot, tensordot_out


def test_tensordot_scalar_dims_matches_aten_two_shapes():
    # dims=int contracts the last axis of both operands.
    torch.manual_seed(0)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    out = tensordot(a, b, 1, 0)
    ref = torch.tensordot(a, b, dims=1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    a2 = torch.randn(33, 17, dtype=torch.float32)
    b2 = torch.randn(17, 5, dtype=torch.float32)
    out2 = tensordot(a2, b2, 1, 0)
    ref2 = torch.tensordot(a2, b2, dims=1)
    torch.testing.assert_close(out2, ref2, rtol=1e-5, atol=1e-5)


def test_tensordot_dim_lists_and_negative_dims_match_aten():
    torch.manual_seed(1)
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    b = torch.randn(3, 4, 5, dtype=torch.float32)
    out = tensordot(a, b, [1, 2], [0, 1])
    ref = torch.tensordot(a, b, dims=([1, 2], [0, 1]))
    assert out.shape == (2, 5)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    outn = tensordot(a, b, [-2, -1], [0, 1])
    torch.testing.assert_close(outn, ref, rtol=1e-5, atol=1e-5)


def test_tensordot_size1_contracted_dim_broadcasts_like_aten():
    # A size-1 contracted axis on one side broadcasts to the other side's
    # extent, exactly like aten: (2,1,3) x (5,3) -> (2,3,3).
    torch.manual_seed(2)
    a = torch.randn(2, 1, 3, dtype=torch.float32)
    b = torch.randn(5, 3, dtype=torch.float32)
    out = tensordot(a, b, [1], [0])
    ref = torch.tensordot(a, b, dims=([1], [0]))
    assert out.shape == (2, 3, 3)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_tensordot_full_contraction_is_scalar():
    torch.manual_seed(3)
    a = torch.randn(3, 4, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    out = tensordot(a, b, [0, 1], [0, 1])
    ref = torch.tensordot(a, b, dims=([0, 1], [0, 1]))
    assert out.shape == ()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_tensordot_dims_zero_is_outer_product():
    torch.manual_seed(4)
    a = torch.arange(3, dtype=torch.float32)
    b = torch.arange(4, dtype=torch.float32)
    out = tensordot(a, b, [], [])
    ref = torch.tensordot(a, b, dims=0)
    assert out.shape == (3, 4)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_tensordot_arange_exact_all_float_dtypes():
    a = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 4
    b = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 5
    torch.testing.assert_close(
        tensordot(a, b, [1, 2], [0, 1]),
        torch.tensordot(a, b, dims=([1, 2], [0, 1])),
        rtol=0,
        atol=0,
    )
    a64 = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4) / 4
    b64 = torch.arange(3 * 4 * 5, dtype=torch.float64).reshape(3, 4, 5) / 5
    torch.testing.assert_close(
        tensordot(a64, b64, [1, 2], [0, 1]),
        torch.tensordot(a64, b64, dims=([1, 2], [0, 1])),
        rtol=0,
        atol=0,
    )


def test_tensordot_fp64_randn():
    # Both sides accumulate in fp64; only the summation order differs.
    # rtol/atol 1e-9 measured against aten fp64 randn.
    torch.manual_seed(5)
    a = torch.randn(6, 7, dtype=torch.float64)
    b = torch.randn(7, 5, dtype=torch.float64)
    out = tensordot(a, b, 1, 0)
    torch.testing.assert_close(out, torch.tensordot(a, b, dims=1), rtol=1e-9, atol=1e-9)


def test_tensordot_zero_k_is_zeros_like_aten():
    a = torch.empty(2, 0, dtype=torch.float32)
    b = torch.empty(0, 3, dtype=torch.float32)
    out = tensordot(a, b, 1, 0)
    ref = torch.tensordot(a, b, dims=1)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_tensordot_non_contiguous_input():
    torch.manual_seed(6)
    base = torch.randn(3, 2, dtype=torch.float32)
    a = base.t()  # (2, 3) non-contiguous view
    assert not a.is_contiguous()
    b = torch.randn(3, 4, dtype=torch.float32)
    out = tensordot(a, b, 1, 0)
    ref = torch.tensordot(a, b, dims=1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_tensordot_errors_match_upstream_validation():
    torch.manual_seed(7)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    with pytest.raises(
        RuntimeError, match="tensordot: number of contracted dims must match"
    ):
        tensordot(a, b, [0], [0, 1])
    with pytest.raises(
        RuntimeError, match="tensordot: contracted dimensions need to match"
    ):
        tensordot(torch.randn(2, 3), torch.randn(2, 4), 1, 0)
    with pytest.raises(IndexError, match="Dimension out of range"):
        tensordot(a, b, [5], [0])
    with pytest.raises(
        RuntimeError, match="tensordot: expected self and other to have the same dtype"
    ):
        tensordot(a, b.to(torch.float64), 1, 0)
    with pytest.raises(RuntimeError, match="tensordot: unsupported dtype"):
        tensordot(a.long(), b.long(), 1, 0)


def test_tensordot_out_writes_buffer():
    torch.manual_seed(8)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    out = torch.empty(2, 4, dtype=torch.float32)
    ret = tensordot_out(a, b, 1, 0, out=out)
    assert ret is out
    torch.testing.assert_close(out, torch.tensordot(a, b, dims=1), rtol=1e-5, atol=1e-5)


def test_tensordot_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.tensordot")
    launches = []
    kernel = mod.tensordot_mm_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(9)
    a = torch.randn(4, 8, dtype=torch.float32)
    b = torch.randn(8, 3, dtype=torch.float32)
    out = tensordot(a, b, 1, 0)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.tensordot(a, b, dims=1), rtol=1e-5, atol=1e-5)
