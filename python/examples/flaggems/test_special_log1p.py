import importlib

import pytest
import torch

from .special_log1p import special_log1p, special_log1p_out


def _tolerance(dtype):
    # Tolerance source: the kernel uses a Taylor branch near zero and
    # log(1 + x) elsewhere in fp32, so the near-zero band is the accuracy
    # critical one; fp32/fp64 stay at 1e-6 relative, fp16 narrows once
    # (1e-3).
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-6, 1e-6


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_log1p(dtype, size):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=dtype) * 20.0 - 0.9

    ref = torch.special.log1p(x)
    tri = special_log1p(x)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_special_log1p_near_zero():
    # log(1 + x) loses every significant digit to cancellation for tiny x;
    # the Taylor branch exists for exactly this band.
    x = torch.tensor(
        [1e-8, -1e-8, 1e-6, -1e-6, 1e-4, -1e-4, 1e-3, -1e-3], dtype=torch.float32
    )

    ref = torch.special.log1p(x)
    tri = special_log1p(x)

    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-9)


def test_special_log1p_special_values():
    # ATen contract probed on torch CPU fp32: log1p(-1) = -inf,
    # x < -1 = nan, +inf stays +inf, -inf = nan, ±0 maps to ±0.
    x = torch.tensor([-1.0, -2.0, float("inf"), float("-inf"), 0.0, -0.0, float("nan")])

    ref = torch.special.log1p(x)
    tri = special_log1p(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == float("-inf") and tri[2].item() == float("inf")


def test_special_log1p_out_variant():
    x = torch.randn(256, dtype=torch.float32).clamp(min=-0.9)
    out = torch.empty_like(x)

    res = special_log1p_out(x, out)
    assert res is out
    torch.testing.assert_close(out, torch.special.log1p(x), rtol=1e-6, atol=1e-6)


def test_special_log1p_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_log1p")
    launches = []
    real_run = mod.special_log1p_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_log1p_func, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 4.0 - 0.5
    out = mod.special_log1p(x)
    assert len(launches) >= 1, "special_log1p must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.log1p(x), rtol=1e-6, atol=1e-6)
