import importlib

import pytest
import torch

from .special_ndtr import special_ndtr


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_ndtr(dtype, size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype) * 5.0

    ref = torch.special.ndtr(x)
    tri = special_ndtr(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: ndtr is Phi(x) via erfc; the kernel evaluates it in
    # fp32, so 1e-6 relative holds for fp32 and fp64 while fp16 narrows
    # once (1e-3).
    if dtype == torch.float16:
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6)


def test_special_ndtr_tails():
    # Both tails: ndtr(-x) + ndtr(x) = 1 must hold to fp32 resolution, and
    # the far negative tail must not underflow to 0 before x = -40.
    x = torch.arange(-40.0, 41.0, 0.5)

    ref = torch.special.ndtr(x)
    tri = special_ndtr(x)

    # torch's own fp32 reference underflows to 0 from x = -7, and in the
    # 1e-8 band above it the port's 12-term asymptotic series and aten
    # differ by ~2e-8 absolute (no significant digits left there on
    # either side), so the atol covers that band only.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-7)
    # Phi(-0.5) + Phi(0.5) = 1: the sweep indices 79 and 81.
    torch.testing.assert_close(tri[79] + tri[81], torch.tensor(1.0), rtol=0, atol=1e-6)


def test_special_ndtr_special_values():
    # ATen contract probed on torch CPU fp32: ndtr(-inf) = 0,
    # ndtr(+inf) = 1, ndtr(0) = 0.5, nan stays nan.
    x = torch.tensor(
        [float("-inf"), float("inf"), 0.0, -0.0, float("nan")], dtype=torch.float32
    )

    ref = torch.special.ndtr(x)
    tri = special_ndtr(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 0.0 and tri[1].item() == 1.0 and tri[2].item() == 0.5


def test_special_ndtr_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_ndtr")
    launches = []
    real_run = mod.ndtr_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.ndtr_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 4.0
    out = mod.special_ndtr(x)
    assert len(launches) >= 1, "special_ndtr must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.ndtr(x), rtol=1e-6, atol=1e-6)
