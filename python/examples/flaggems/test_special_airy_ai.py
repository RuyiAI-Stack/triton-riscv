import importlib

import pytest
import torch

from .special_airy_ai import special_airy_ai, special_airy_ai_out


@pytest.mark.parametrize("size", [512, 1023])
def test_special_airy_ai(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 6.0

    ref = torch.special.airy_ai(x)
    tri = special_airy_ai(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: Ai(x) is the Cephes rational approximation below
    # 7.25 and the exponentially scaled asymptotic form above, fp32
    # throughout; 1e-5 relative covers both branches, and the atol covers
    # the zero crossing at -2.3381 and the far-negative oscillations.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_airy_ai_branches():
    # Both branches: decay for x > 0, oscillation for x < 0, and the
    # classic values at 0, 1 and -1.
    x = torch.cat(
        [
            torch.linspace(-20.0, 20.0, 401),
            torch.tensor([0.0, 1.0, -1.0, 7.25, 7.2500001, 12.0, 18.0]),
        ]
    )

    ref = torch.special.airy_ai(x)
    tri = special_airy_ai(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tri[401], torch.tensor(0.3550281), rtol=1e-5, atol=1e-6)


def test_special_airy_ai_special_values():
    # ATen contract probed on torch CPU fp32: Ai at either infinity is nan
    # on this torch build, and nan stays nan.
    x = torch.tensor([float("inf"), float("-inf"), float("nan")], dtype=torch.float32)

    ref = torch.special.airy_ai(x)
    tri = special_airy_ai(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_airy_ai_out_variant():
    x = torch.randn(256, dtype=torch.float32) * 3.0
    out = torch.empty_like(x)

    res = special_airy_ai_out(x, out=out)
    assert res is out
    torch.testing.assert_close(out, torch.special.airy_ai(x), rtol=1e-5, atol=1e-6)


def test_special_airy_ai_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_airy_ai")
    launches = []
    real_run = mod.special_airy_ai_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_airy_ai_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 4.0
    out = mod.special_airy_ai(x)
    assert len(launches) >= 1, "special_airy_ai must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.airy_ai(x), rtol=1e-5, atol=1e-6)
