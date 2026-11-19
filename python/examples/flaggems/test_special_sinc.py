import importlib

import pytest
import torch

from .special_sinc import special_sinc


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_sinc(dtype, size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype) * 6.0

    ref = torch.special.sinc(x)
    tri = special_sinc(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: sinc = sin(pi x) / (pi x) evaluated in fp32; the
    # zeros of sin land on ~1e-8 rather than 0 (probed: torch returns
    # -2.78e-8 at x = 1), so the atol dominates there; fp16 narrows once.
    if dtype == torch.float16:
        torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-3)
    else:
        torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-7)


def test_special_sinc_zero_and_near_zero():
    # sinc(0) = 1 exactly (the kernel special-cases the pole), and the
    # first Taylor terms must stay intact immediately around it.
    x = torch.tensor([0.0, -0.0, 1e-7, -1e-7, 1e-4, -1e-4, 1e-2], dtype=torch.float32)

    ref = torch.special.sinc(x)
    tri = special_sinc(x)

    assert tri[0].item() == 1.0 and tri[1].item() == 1.0
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-9)


def test_special_sinc_special_values():
    x = torch.tensor([float("inf"), float("-inf"), float("nan")])

    ref = torch.special.sinc(x)
    tri = special_sinc(x)

    assert bool(tri.isnan().all()), "sinc at non-finite input is nan on torch CPU"
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_sinc_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_sinc")
    launches = []
    real_run = mod.special_sinc_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_sinc_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.special_sinc(x)
    assert len(launches) >= 1, "special_sinc must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.sinc(x), rtol=1e-6, atol=1e-7)
