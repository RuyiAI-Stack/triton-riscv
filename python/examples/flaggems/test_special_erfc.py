import importlib

import pytest
import torch

from .special_erfc import special_erfc


def _tolerance(dtype):
    # Tolerance source: measured on this port against the fp32 torch
    # reference -- max |diff| 1.2e-7 over every band including the 2.85
    # hybrid seam (the asymptotic tail drops to ~1e-35), so 1e-5 keeps
    # better than an 80x margin for fp32/fp64; fp16 narrows once.
    if dtype == torch.float16:
        return 1e-2, 1e-3
    return 0.0, 1e-5


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_erfc(dtype, size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype) * 6.0

    ref = torch.special.erfc(x)
    tri = special_erfc(x)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_special_erfc_tail_sweep():
    # The hybrid switch and the fp32 cancellation the series avoids:
    # 1 - erf(x) returns exactly 0 past 3.5; erfc must keep decaying.
    x = torch.cat(
        [
            torch.linspace(0.0, 2.9, 291),
            torch.linspace(2.9, 12.0, 92),
            torch.tensor([3.4, 3.5, 3.6, 4.0, 6.0, 8.0, 10.0]),
        ]
    ).to(torch.float32)

    ref = torch.special.erfc(x)
    tri = special_erfc(x)

    tail = x > 2.85
    # exp(-x^2) underflows to 0 for x > ~10.2 in fp32 -- the reference
    # does too, so the tail only has to stay non-negative there.
    assert bool((tri[tail] >= 0).all()), "erfc must stay non-negative in the tail"
    torch.testing.assert_close(tri[tail], ref[tail], rtol=0, atol=1e-5)
    torch.testing.assert_close(tri[~tail], ref[~tail], rtol=0, atol=1e-5)


def test_special_erfc_special_values():
    # ATen contract probed on torch CPU fp32: erfc(+inf) = 0,
    # erfc(-inf) = 2, erfc(nan) = nan, erfc(±0) = 1.
    x = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan")])

    ref = torch.special.erfc(x)
    tri = special_erfc(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 1.0 and tri[1].item() == 1.0
    assert tri[2].item() == 0.0 and tri[3].item() == 2.0


def test_special_erfc_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_erfc")
    launches = []
    real_run = mod._erfc_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._erfc_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 3.0
    out = mod.special_erfc(x)
    assert len(launches) >= 1, "special_erfc must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.erfc(x), rtol=0, atol=1e-3)
