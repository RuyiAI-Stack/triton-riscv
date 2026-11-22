import importlib

import pytest
import torch

from .linalg_vecdot import linalg_vecdot, linalg_vecdot_out


def test_vecdot_default_dim_matches_aten():
    torch.manual_seed(0)
    for shape in ((7,), (2, 3), (2, 3, 4)):
        x = torch.randn(*shape, dtype=torch.float32)
        y = torch.randn(*shape, dtype=torch.float32)
        out = linalg_vecdot(x, y)
        ref = torch.linalg.vecdot(x, y)
        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5, msg=f"{shape}")


def test_vecdot_explicit_dim_matches_aten():
    # dim != -1 takes the general movedim path.
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    y = torch.randn(2, 3, 4, dtype=torch.float32)
    for dim in (0, 1, 2, -2, -3):
        out = linalg_vecdot(x, y, dim=dim)
        ref = torch.linalg.vecdot(x, y, dim=dim)
        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5, msg=f"dim={dim}")


def test_vecdot_positional_dim_matches_aten():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    y = torch.randn(2, 3, 4, dtype=torch.float32)
    out = linalg_vecdot(x, y, 1)
    ref = torch.linalg.vecdot(x, y, dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_vecdot_arange_exact_all_dtypes():
    # Integer-valued data makes the reduction exact regardless of block order.
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 4
    y = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 4
    torch.testing.assert_close(
        linalg_vecdot(x, y), torch.linalg.vecdot(x, y), rtol=0, atol=0
    )
    x64 = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4) / 4
    y64 = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4) / 4
    torch.testing.assert_close(
        linalg_vecdot(x64, y64), torch.linalg.vecdot(x64, y64), rtol=0, atol=0
    )
    xi = torch.arange(2 * 3 * 4, dtype=torch.int64).reshape(2, 3, 4) % 5
    yi = torch.arange(2 * 3 * 4, dtype=torch.int64).reshape(2, 3, 4) % 4
    outi = linalg_vecdot(xi, yi)
    assert outi.dtype == torch.int64
    # aten's torch.linalg.vecdot rejects integer inputs; the integer contract
    # (fp32 accumulation of exactly representable products) is pinned against
    # the integer dot directly.
    torch.testing.assert_close(outi, (xi * yi).sum(-1), rtol=0, atol=0)


def test_vecdot_fp64_randn():
    # The upstream kernel accumulates in fp32 even for fp64 inputs
    # (IS_FP64 is not set in _vecdot_kernel), so randn fp64 compares at the
    # fp32 noise level; rtol/atol 1e-4 measured against aten fp64.
    torch.manual_seed(3)
    x = torch.randn(2, 3, 64, dtype=torch.float64)
    y = torch.randn(2, 3, 64, dtype=torch.float64)
    out = linalg_vecdot(x, y)
    torch.testing.assert_close(out, torch.linalg.vecdot(x, y), rtol=1e-4, atol=1e-4)


def test_vecdot_zero_vdim_is_zeros_like_aten():
    x = torch.empty(2, 0, dtype=torch.float32)
    y = torch.empty(2, 0, dtype=torch.float32)
    out = linalg_vecdot(x, y)
    ref = torch.linalg.vecdot(x, y)
    assert out.shape == (2,)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_vecdot_non_contiguous_input():
    torch.manual_seed(4)
    base = torch.randn(4, 3, dtype=torch.float32)
    x = base.t()  # (3, 4) non-contiguous view
    assert not x.is_contiguous()
    y = torch.randn(3, 4, dtype=torch.float32)
    out = linalg_vecdot(x, y)
    ref = torch.linalg.vecdot(x, y)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_vecdot_errors_match_upstream_validation():
    torch.manual_seed(5)
    x = torch.randn(2, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="Input shapes must match"):
        linalg_vecdot(x, torch.randn(2, 4))
    with pytest.raises(IndexError, match="Dimension out of range"):
        linalg_vecdot(x, x, dim=5)


def test_vecdot_out_variant_writes_buffer():
    torch.manual_seed(6)
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    y = torch.randn(2, 3, 4, dtype=torch.float32)
    out = torch.empty(2, 3, dtype=torch.float32)
    ret = linalg_vecdot_out(x, y, out=out)
    assert ret is out
    torch.testing.assert_close(out, torch.linalg.vecdot(x, y), rtol=1e-5, atol=1e-5)


def test_vecdot_out_without_out_computes_fresh():
    torch.manual_seed(7)
    x = torch.randn(5, dtype=torch.float64)
    y = torch.randn(5, dtype=torch.float64)
    out = linalg_vecdot_out(x, y)
    assert out.shape == ()
    torch.testing.assert_close(out, torch.linalg.vecdot(x, y), rtol=1e-4, atol=1e-4)


def test_vecdot_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.linalg_vecdot")
    launches = []
    kernel = mod._vecdot_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(8)
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    y = torch.randn(2, 3, 8, dtype=torch.float32)
    out = linalg_vecdot(x, y)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.linalg.vecdot(x, y), rtol=1e-5, atol=1e-5)
