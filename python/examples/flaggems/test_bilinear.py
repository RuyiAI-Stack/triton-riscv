import importlib

import pytest
import torch
import torch.nn.functional as F

from .bilinear import bilinear


def _bilinear_ref(i1, i2, weight, bias):
    return F.bilinear(i1, i2, weight, bias)


def test_bilinear_matches_aten_two_shapes():
    torch.manual_seed(0)
    for m, k1, k2, n in ((5, 3, 4, 2), (7, 6, 5, 3)):
        i1 = torch.randn(m, k1, dtype=torch.float32)
        i2 = torch.randn(m, k2, dtype=torch.float32)
        weight = torch.randn(n, k1, k2, dtype=torch.float32)
        bias = torch.randn(n, dtype=torch.float32)
        out = bilinear(i1, i2, weight, bias)
        ref = _bilinear_ref(i1, i2, weight, bias)
        torch.testing.assert_close(
            out, ref, rtol=1e-5, atol=1e-5, msg=f"shape {(m, k1, k2, n)}"
        )
        assert out.shape == (m, n)


def test_bilinear_no_bias_matches_aten():
    torch.manual_seed(1)
    i1 = torch.randn(5, 3, dtype=torch.float32)
    i2 = torch.randn(5, 4, dtype=torch.float32)
    weight = torch.randn(2, 3, 4, dtype=torch.float32)
    out = bilinear(i1, i2, weight)
    ref = _bilinear_ref(i1, i2, weight, None)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_bilinear_batched_3d_matches_aten():
    # (N, *, in_features) batch dims flatten to M inside the wrapper.
    torch.manual_seed(2)
    i1 = torch.randn(2, 5, 3, dtype=torch.float32)
    i2 = torch.randn(2, 5, 4, dtype=torch.float32)
    weight = torch.randn(2, 3, 4, dtype=torch.float32)
    bias = torch.randn(2, dtype=torch.float32)
    out = bilinear(i1, i2, weight, bias)
    ref = _bilinear_ref(i1, i2, weight, bias)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert out.shape == (2, 5, 2)


def test_bilinear_fp64_arange_exact():
    # Dyadic divisors keep every value, product and partial sum exactly
    # representable in the wrapper's fp32 accumulation, so fp64 matches aten
    # bit for bit regardless of the codegen's contraction order.
    i1 = torch.arange(15, dtype=torch.float64).reshape(5, 3) / 2
    i2 = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 4
    weight = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 8
    bias = torch.arange(2, dtype=torch.float64)
    out = bilinear(i1, i2, weight, bias)
    ref = _bilinear_ref(i1, i2, weight, bias)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.float64


def test_bilinear_fp64_randn():
    # The upstream kernel always accumulates in fp32 (loads are cast), so a
    # randn fp64 comparison runs at the fp32 noise level; 1e-4 measured
    # against aten fp64.
    torch.manual_seed(3)
    i1 = torch.randn(6, 3, dtype=torch.float64)
    i2 = torch.randn(6, 4, dtype=torch.float64)
    weight = torch.randn(2, 3, 4, dtype=torch.float64)
    bias = torch.randn(2, dtype=torch.float64)
    out = bilinear(i1, i2, weight, bias)
    ref = _bilinear_ref(i1, i2, weight, bias)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_bilinear_int64_small_values_exact():
    i1 = torch.arange(15, dtype=torch.int64).reshape(5, 3) % 5
    i2 = torch.arange(20, dtype=torch.int64).reshape(5, 4) % 3
    weight = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4) % 4
    bias = torch.arange(2, dtype=torch.int64)
    out = bilinear(i1, i2, weight, bias)
    ref = _bilinear_ref(i1, i2, weight, bias)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.int64


def test_bilinear_empty_batch():
    torch.manual_seed(4)
    i1 = torch.empty(0, 3, dtype=torch.float32)
    i2 = torch.empty(0, 4, dtype=torch.float32)
    weight = torch.randn(2, 3, 4, dtype=torch.float32)
    bias = torch.randn(2, dtype=torch.float32)
    out = bilinear(i1, i2, weight, bias)
    ref = _bilinear_ref(i1, i2, weight, bias)
    assert out.shape == (0, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_bilinear_non_contiguous_input():
    torch.manual_seed(5)
    base = torch.randn(3, 5, dtype=torch.float32)
    i1 = base.t()  # (5, 3) non-contiguous view
    assert not i1.is_contiguous()
    i2 = torch.randn(5, 4, dtype=torch.float32)
    weight = torch.randn(2, 3, 4, dtype=torch.float32)
    bias = torch.randn(2, dtype=torch.float32)
    out = bilinear(i1, i2, weight, bias)
    ref = _bilinear_ref(i1, i2, weight, bias)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_bilinear_errors_match_aten():
    # aten's own bilinear validation, pinned empirically.
    torch.manual_seed(6)
    i1 = torch.randn(5, 3, dtype=torch.float32)
    i2 = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(
        RuntimeError, match=r"bilinear\(\): input1 size does not match weight size"
    ):
        bilinear(i1, i2, torch.randn(2, 2, 4))
    with pytest.raises(
        RuntimeError, match=r"bilinear\(\): bias size does not match weight size"
    ):
        bilinear(i1, i2, torch.randn(2, 3, 4), torch.randn(3))


def test_bilinear_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.bilinear")
    launches = []
    kernel = mod.bilinear_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(7)
    i1 = torch.randn(5, 3, dtype=torch.float32)
    i2 = torch.randn(5, 4, dtype=torch.float32)
    weight = torch.randn(2, 3, 4, dtype=torch.float32)
    bias = torch.randn(2, dtype=torch.float32)
    out = bilinear(i1, i2, weight, bias)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out, _bilinear_ref(i1, i2, weight, bias), rtol=1e-5, atol=1e-5
    )
