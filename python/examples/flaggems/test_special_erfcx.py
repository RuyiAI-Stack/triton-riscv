import importlib

import pytest
import torch

from .special_erfcx import special_erfcx


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_erfcx(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 5.0

    ref = torch.special.erfcx(x)
    tri = special_erfcx(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: erfcx = exp(x^2) * erfc(x); the kernel evaluates
    # both branches in fp32. 1e-5 relative covers x in [-20, 20], where
    # erfcx spans ~1e-175 to ~2e174; the atol covers the zero crossing
    # that erfcx inherits from erfc for large positive x.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_erfcx_large_arguments():
    # Large positive x decays like 1 / (sqrt(pi) x) and must not underflow
    # early; large negative x grows like 2 exp(x^2) and must stay finite
    # until fp32 overflows.
    x = torch.tensor(
        [-20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 26.0],
        dtype=torch.float32,
    )

    ref = torch.special.erfcx(x)
    tri = special_erfcx(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-6)
    # the negative rows stay huge (2 e^{x^2}) instead of collapsing to
    # the tiny positive-side values: -5 alone is ~1.4e11.
    assert tri[2] > 1e10


def test_special_erfcx_special_values():
    # ATen contract probed on torch CPU fp32: erfcx(+inf) = 0,
    # erfcx(-inf) = +inf, erfcx(nan) = nan.
    x = torch.tensor(
        [0.0, float("inf"), float("-inf"), float("nan")], dtype=torch.float32
    )

    ref = torch.special.erfcx(x)
    tri = special_erfcx(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 1.0
    assert tri[1].item() == 0.0 and tri[2].item() == float("inf")


def test_special_erfcx_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_erfcx")
    launches = []
    real_run = mod.special_erfcx_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_erfcx_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 4.0
    out = mod.special_erfcx(x)
    assert len(launches) >= 1, "special_erfcx must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.erfcx(x), rtol=1e-5, atol=1e-6)
