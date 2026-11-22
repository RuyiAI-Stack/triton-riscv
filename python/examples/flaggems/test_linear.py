import importlib

import pytest
import torch
import torch.nn.functional as F

from .linear import linear


def test_linear_matches_aten_two_shapes():
    torch.manual_seed(0)
    for m, k, n in ((8, 16, 32), (33, 17, 65)):
        x = torch.randn(m, k, dtype=torch.float32)
        w = torch.randn(n, k, dtype=torch.float32)
        bias = torch.randn(n, dtype=torch.float32)
        out = linear(x, w, bias)
        ref = F.linear(x, w, bias)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5, msg=f"{(m, k, n)}")
        assert out.shape == (m, n)


def test_linear_no_bias_matches_aten():
    torch.manual_seed(1)
    x = torch.randn(8, 16, dtype=torch.float32)
    w = torch.randn(32, 16, dtype=torch.float32)
    out = linear(x, w)
    ref = F.linear(x, w)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_linear_gemv_path_small_m_matches_aten():
    # fp32 inputs with M <= 5 take the upstream GEMV split-K path.
    torch.manual_seed(2)
    for m in (1, 2, 5):
        x = torch.randn(m, 16, dtype=torch.float32)
        w = torch.randn(32, 16, dtype=torch.float32)
        bias = torch.randn(32, dtype=torch.float32)
        out = linear(x, w, bias)
        ref = F.linear(x, w, bias)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5, msg=f"m={m}")


def test_linear_batched_3d_and_1d_match_aten():
    torch.manual_seed(3)
    w = torch.randn(4, 5, dtype=torch.float32)
    bias = torch.randn(4, dtype=torch.float32)
    x3 = torch.randn(2, 3, 5, dtype=torch.float32)
    out3 = linear(x3, w, bias)
    ref3 = F.linear(x3, w, bias)
    assert out3.shape == (2, 3, 4)
    torch.testing.assert_close(out3, ref3, rtol=1e-5, atol=1e-5)
    x1 = torch.randn(5, dtype=torch.float32)
    out1 = linear(x1, w, bias)
    ref1 = F.linear(x1, w, bias)
    assert out1.shape == (4,)
    torch.testing.assert_close(out1, ref1, rtol=1e-5, atol=1e-5)


def test_linear_fp64_arange_exact():
    # Integer-valued data keeps the kernel's fp32 accumulation exact.
    x = torch.arange(4 * 8, dtype=torch.float64).reshape(4, 8) / 8
    w = torch.arange(6 * 8, dtype=torch.float64).reshape(6, 8) / 8
    bias = torch.arange(6, dtype=torch.float64)
    out = linear(x, w, bias)
    ref = F.linear(x, w, bias)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.float64


def test_linear_fp64_randn():
    # The upstream kernel accumulates in fp32 regardless of dtype; rtol/atol
    # 1e-4 measured against aten fp64 randn.
    torch.manual_seed(4)
    x = torch.randn(8, 16, dtype=torch.float64)
    w = torch.randn(32, 16, dtype=torch.float64)
    bias = torch.randn(32, dtype=torch.float64)
    out = linear(x, w, bias)
    ref = F.linear(x, w, bias)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_linear_int64_small_values_exact():
    x = torch.arange(4 * 8, dtype=torch.int64).reshape(4, 8) % 5
    w = torch.arange(6 * 8, dtype=torch.int64).reshape(6, 8) % 4
    bias = torch.arange(6, dtype=torch.int64)
    out = linear(x, w, bias)
    ref = F.linear(x, w, bias)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.int64


def test_linear_empty_batch():
    torch.manual_seed(5)
    x = torch.empty(0, 16, dtype=torch.float32)
    w = torch.randn(32, 16, dtype=torch.float32)
    bias = torch.randn(32, dtype=torch.float32)
    out = linear(x, w, bias)
    ref = F.linear(x, w, bias)
    assert out.shape == (0, 32)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_linear_non_contiguous_weight_view():
    torch.manual_seed(6)
    base = torch.randn(16, 32, dtype=torch.float32)
    w = base.t()  # (32, 16) non-contiguous view
    assert not w.is_contiguous()
    x = torch.randn(8, 16, dtype=torch.float32)
    bias = torch.randn(32, dtype=torch.float32)
    out = linear(x, w, bias)
    ref = F.linear(x, w, bias)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_linear_non_contiguous_input_view():
    torch.manual_seed(7)
    base = torch.randn(16, 8, dtype=torch.float32)
    x = base.t()  # (8, 16) non-contiguous view
    assert not x.is_contiguous()
    w = torch.randn(32, 16, dtype=torch.float32)
    out = linear(x, w)
    ref = F.linear(x, w)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_linear_error_matches_upstream_validation():
    torch.manual_seed(8)
    x = torch.randn(5, 3, dtype=torch.float32)
    w = torch.randn(4, 2, dtype=torch.float32)
    with pytest.raises(
        RuntimeError, match=r"weight \(4, 2\) is not an \(N, 3\) matrix"
    ):
        linear(x, w)


def test_linear_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.linear")
    launches = []
    kernel = mod.linear_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(9)
    # M=8 keeps the call on the tiled-dot path the main kernel serves.
    x = torch.randn(8, 16, dtype=torch.float32)
    w = torch.randn(32, 16, dtype=torch.float32)
    bias = torch.randn(32, dtype=torch.float32)
    out = linear(x, w, bias)
    assert len(launches) >= 1
    torch.testing.assert_close(out, F.linear(x, w, bias), rtol=1e-5, atol=1e-5)
