import importlib

import pytest
import torch

from .addbmm import addbmm, addbmm_


def test_addbmm_matches_aten_fp32_two_shapes():
    torch.manual_seed(0)
    for b, m, k, n in ((2, 16, 16, 16), (3, 33, 17, 65)):
        bias = torch.randn(m, n, dtype=torch.float32)
        batch1 = torch.randn(b, m, k, dtype=torch.float32)
        batch2 = torch.randn(b, k, n, dtype=torch.float32)
        out = addbmm(bias, batch1, batch2)
        ref = torch.addbmm(bias, batch1, batch2)
        torch.testing.assert_close(
            out, ref, rtol=1e-5, atol=1e-5, msg=f"shape {(b, m, k, n)}"
        )
        assert out.dtype == torch.float32 and out.shape == (m, n)


def test_addbmm_alpha_beta_match_aten():
    torch.manual_seed(1)
    bias = torch.randn(4, 5, dtype=torch.float32)
    batch1 = torch.randn(2, 4, 3, dtype=torch.float32)
    batch2 = torch.randn(2, 3, 5, dtype=torch.float32)
    out = addbmm(bias, batch1, batch2, beta=0.5, alpha=2.0)
    ref = torch.addbmm(bias, batch1, batch2, beta=0.5, alpha=2.0)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_addbmm_broadcast_bias_one_row_matches_aten():
    # aten broadcasts a (1, N) bias across the (B, M, N) batch product.
    torch.manual_seed(2)
    batch1 = torch.randn(2, 4, 3, dtype=torch.float32)
    batch2 = torch.randn(2, 3, 5, dtype=torch.float32)
    bias = torch.randn(1, 5, dtype=torch.float32).expand(4, 5)
    out = addbmm(bias, batch1, batch2)
    ref = torch.addbmm(bias, batch1, batch2)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_addbmm_fp64_arange_exact():
    # Integer-valued data keeps the fp32 operand cast inside the upstream
    # kernel exact, so fp64 must match aten bit for bit.
    bias = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    batch1 = torch.arange(3 * 3 * 4, dtype=torch.float64).reshape(3, 3, 4) / 4
    batch2 = torch.arange(3 * 4 * 4, dtype=torch.float64).reshape(3, 4, 4) / 2
    out = addbmm(bias, batch1, batch2, beta=2.0, alpha=3.0)
    ref = torch.addbmm(bias, batch1, batch2, beta=2.0, alpha=3.0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_addbmm_fp64_randn():
    # The upstream kernel rounds the matmul operands to fp32 (IS_FP64 branch
    # casts a/b before tl.dot); 1e-5 measured against aten fp64 randn inputs.
    torch.manual_seed(3)
    bias = torch.randn(8, 8, dtype=torch.float64)
    batch1 = torch.randn(2, 8, 8, dtype=torch.float64)
    batch2 = torch.randn(2, 8, 8, dtype=torch.float64)
    out = addbmm(bias, batch1, batch2)
    ref = torch.addbmm(bias, batch1, batch2)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_addbmm_int64_small_values_exact():
    bias = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    batch1 = torch.arange(2 * 2 * 4, dtype=torch.int64).reshape(2, 2, 4) % 7
    batch2 = torch.arange(2 * 4 * 3, dtype=torch.int64).reshape(2, 4, 3) % 5
    out = addbmm(bias, batch1, batch2)
    ref = torch.addbmm(bias, batch1, batch2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == torch.int64


def test_addbmm_empty_batch_is_beta_times_bias():
    torch.manual_seed(4)
    bias = torch.randn(3, 4, dtype=torch.float32)
    out = addbmm(bias, torch.empty(0, 3, 5), torch.empty(0, 5, 4), beta=2.0, alpha=3.0)
    ref = torch.addbmm(
        bias, torch.empty(0, 3, 5), torch.empty(0, 5, 4), beta=2.0, alpha=3.0
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (3, 4)


def test_addbmm_zero_k_matches_aten():
    torch.manual_seed(5)
    bias = torch.randn(3, 4, dtype=torch.float32)
    out = addbmm(bias, torch.empty(2, 3, 0), torch.empty(2, 0, 4))
    ref = torch.addbmm(bias, torch.empty(2, 3, 0), torch.empty(2, 0, 4))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_addbmm_non_contiguous_batch():
    torch.manual_seed(6)
    base = torch.randn(2, 4, 3, dtype=torch.float32)
    batch1 = base.transpose(1, 2)  # (2, 3, 4) non-contiguous view
    assert not batch1.is_contiguous()
    bias = torch.randn(3, 4, dtype=torch.float32)
    batch2 = torch.randn(2, 4, 4, dtype=torch.float32)
    out = addbmm(bias, batch1, batch2)
    ref = torch.addbmm(bias, batch1, batch2)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_addbmm_inplace_writes_into_bias():
    torch.manual_seed(7)
    bias = torch.randn(4, 5, dtype=torch.float32)
    orig = bias.clone()  # snapshot before the in-place call clobbers it
    batch1 = torch.randn(2, 4, 3, dtype=torch.float32)
    batch2 = torch.randn(2, 3, 5, dtype=torch.float32)
    ret = addbmm_(bias, batch1, batch2, beta=0.5, alpha=2.0)
    assert ret is bias
    ref = torch.addbmm(orig, batch1, batch2, beta=0.5, alpha=2.0)
    torch.testing.assert_close(bias, ref, rtol=1e-5, atol=1e-5)


def test_addbmm_errors_match_upstream_validation():
    torch.manual_seed(8)
    bias = torch.randn(3, 4)
    batch1 = torch.randn(2, 3, 5)
    batch2 = torch.randn(2, 5, 4)
    with pytest.raises(AssertionError, match="batch1 and batch2 must be 3D"):
        addbmm(bias, torch.randn(3, 5), batch2)
    with pytest.raises(AssertionError, match="batch1 and batch2 must be 3D"):
        addbmm(bias, batch1, torch.randn(5, 4))
    with pytest.raises(AssertionError, match="bias must be 2D"):
        addbmm(torch.randn(2, 3, 4), batch1, batch2)
    with pytest.raises(AssertionError, match="Inner dimensions must match"):
        addbmm(bias, torch.randn(2, 3, 6), batch2)
    with pytest.raises(AssertionError, match=r"bias shape mismatch: expected \(3, 4\)"):
        addbmm(torch.randn(3, 5), batch1, batch2)


def test_addbmm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.addbmm")
    launches = []
    kernel = mod.addbmm_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(9)
    bias = torch.randn(4, 5, dtype=torch.float32)
    batch1 = torch.randn(2, 4, 3, dtype=torch.float32)
    batch2 = torch.randn(2, 3, 5, dtype=torch.float32)
    out = addbmm(bias, batch1, batch2)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out, torch.addbmm(bias, batch1, batch2), rtol=1e-5, atol=1e-5
    )
