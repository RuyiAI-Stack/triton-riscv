import importlib

import pytest
import torch

from .chain_matmul import chain_matmul, chain_matmul_out


def test_chain_matmul_two_and_three_matches_aten():
    torch.manual_seed(0)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    c = torch.randn(4, 5, dtype=torch.float32)
    out = chain_matmul([a, b])
    ref = torch.chain_matmul(a, b)  # deprecated oracle, warning is fine
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    out3 = chain_matmul([a, b, c])
    ref3 = torch.chain_matmul(a, b, c)
    torch.testing.assert_close(out3, ref3, rtol=1e-5, atol=1e-5)
    assert out3.shape == (2, 5)


def test_chain_matmul_list_and_varargs_forms_agree():
    torch.manual_seed(1)
    mats = [
        torch.randn(16, 32, dtype=torch.float32),
        torch.randn(32, 8, dtype=torch.float32),
        torch.randn(8, 64, dtype=torch.float32),
        torch.randn(64, 4, dtype=torch.float32),
    ]
    out_list = chain_matmul(mats)
    out_vararg = chain_matmul(*mats)
    ref = torch.chain_matmul(*mats)
    torch.testing.assert_close(out_list, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_vararg, ref, rtol=1e-5, atol=1e-5)


def test_chain_matmul_four_dp_order_matches_aten():
    # Rectangular chain where the minimum-flop parenthesization matters.
    torch.manual_seed(2)
    mats = [
        torch.randn(8, 64, dtype=torch.float32),
        torch.randn(64, 4, dtype=torch.float32),
        torch.randn(4, 128, dtype=torch.float32),
        torch.randn(128, 2, dtype=torch.float32),
    ]
    out = chain_matmul(mats)
    ref = torch.chain_matmul(*mats)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_chain_matmul_fp64_arange_exact():
    mats = [
        torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3) / 3,
        torch.arange(3 * 4, dtype=torch.float64).reshape(3, 4) / 4,
        torch.arange(4 * 5, dtype=torch.float64).reshape(4, 5) / 5,
    ]
    out = chain_matmul(mats)
    ref = torch.chain_matmul(*mats)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_chain_matmul_fp64_randn():
    # Full-fp64 accumulation on both sides; only the summation order differs
    # from aten. rtol/atol 1e-9 measured against aten fp64 randn chains.
    torch.manual_seed(3)
    mats = [
        torch.randn(6, 6, dtype=torch.float64),
        torch.randn(6, 7, dtype=torch.float64),
        torch.randn(7, 5, dtype=torch.float64),
    ]
    out = chain_matmul(mats)
    ref = torch.chain_matmul(*mats)
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)


def test_chain_matmul_single_matrix_returns_copy():
    torch.manual_seed(4)
    a = torch.randn(3, 4, dtype=torch.float64)
    out = chain_matmul([a])
    ref = torch.chain_matmul(a)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.data_ptr() != a.data_ptr()


def test_chain_matmul_empty_matrix_matches_aten():
    torch.manual_seed(5)
    a = torch.empty(0, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    out = chain_matmul([a, b])
    ref = torch.chain_matmul(a, b)
    assert out.shape == (0, 4)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_chain_matmul_non_contiguous_operand():
    torch.manual_seed(6)
    base = torch.randn(3, 2, dtype=torch.float32)
    a = base.t()  # (2, 3) non-contiguous view
    assert not a.is_contiguous()
    b = torch.randn(3, 4, dtype=torch.float32)
    out = chain_matmul([a, b])
    ref = torch.chain_matmul(a, b)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_chain_matmul_errors_match_upstream_validation():
    torch.manual_seed(7)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Tensor dimension is 3, expected 2 instead"):
        chain_matmul([torch.randn(2, 3, 4), b])
    with pytest.raises(
        RuntimeError, match=r"multi_dot\(\): all tensors must have the same dtype"
    ):
        chain_matmul([a, torch.randn(3, 4, dtype=torch.float64)])
    with pytest.raises(
        RuntimeError, match=r"multi_dot\(\): tensors cannot be multiplied"
    ):
        chain_matmul([a, torch.randn(5, 6)])
    with pytest.raises(
        RuntimeError, match=r"chain_matmul\(\): Expected one or more matrices"
    ):
        chain_matmul([])
    with pytest.raises(RuntimeError, match=r"chain_matmul\(\): unsupported dtype"):
        # same-dtype non-float pair: a mixed pair hits upstream's dtype/device
        # mismatch check first ("all tensors must have the same dtype and
        # device"), which fires before the floating-point check.
        chain_matmul([torch.arange(6).reshape(2, 3), torch.arange(12).reshape(3, 4)])


def test_chain_matmul_out_writes_buffer():
    torch.manual_seed(8)
    mats = [
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(3, 4, dtype=torch.float32),
        torch.randn(4, 5, dtype=torch.float32),
    ]
    out = torch.empty(2, 5, dtype=torch.float32)
    ret = chain_matmul_out(mats, out=out)
    assert ret is out
    ref = torch.chain_matmul(*mats)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_chain_matmul_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.chain_matmul")
    launches = []
    kernel = mod._gemm  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(9)
    mats = [
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(3, 4, dtype=torch.float32),
        torch.randn(4, 5, dtype=torch.float32),
    ]
    out = chain_matmul(mats)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.chain_matmul(*mats), rtol=1e-5, atol=1e-5)
