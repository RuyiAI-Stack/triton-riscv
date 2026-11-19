import importlib

import pytest
import torch

from .special_expit import special_expit


def _tolerance(dtype):
    # Tolerance source: expit is 1 / (1 + exp(-x)); the kernel computes the
    # closed form in fp32 and narrows once, so fp32/fp64 stay at a couple
    # of exp ulp (1e-6) and fp16 rounds the same fp32 value (1e-3).
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-6, 1e-6


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_expit(dtype, size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype) * 12.0

    ref = torch.special.expit(x)
    tri = special_expit(x)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_special_expit_saturation():
    # exp() overflows for |x| > 88 in fp32; the closed form must saturate
    # to 1 / 0 on both sides instead of producing inf / nan.
    x = torch.tensor(
        [-200.0, -100.0, -88.0, -20.0, 0.0, 20.0, 88.0, 100.0, 200.0],
        dtype=torch.float32,
    )

    ref = torch.special.expit(x)
    tri = special_expit(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=1e-6)
    assert tri[0].item() == 0.0 and tri[-1].item() == 1.0


def test_special_expit_special_values():
    x = torch.tensor(
        [float("-inf"), float("inf"), float("nan"), -0.0], dtype=torch.float32
    )

    ref = torch.special.expit(x)
    tri = special_expit(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 0.0 and tri[1].item() == 1.0 and tri[3].item() == 0.5


def test_special_expit_non_contiguous():
    base = torch.randn(2048, dtype=torch.float32) * 4.0
    x = base[::2]

    assert not x.is_contiguous()
    torch.testing.assert_close(
        special_expit(x), torch.special.expit(x.clone()), rtol=1e-6, atol=1e-6
    )


def test_special_expit_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_expit")
    launches = []
    real_run = mod.special_expit_forward.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_expit_forward, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 4.0
    out = mod.special_expit(x)
    assert len(launches) >= 1, "special_expit must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.expit(x), rtol=1e-6, atol=1e-6)
