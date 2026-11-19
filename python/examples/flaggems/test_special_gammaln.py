import importlib

import pytest
import torch

from .special_gammaln import special_gammaln, special_gammaln_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_gammaln(size):
    torch.manual_seed(0)
    # Keep away from the poles at 0 and the negative integers.
    x = torch.rand(size, dtype=torch.float32) * 30.0 + 0.01

    ref = torch.special.gammaln(x)
    tri = special_gammaln(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: gammaln uses the Lanczos approximation (g = 7) with
    # reflection, the same construction measured in the lgamma port at a
    # 6.4e-4 worst case including the x < 0.5 branch; 1e-3 keeps margin.
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)


def test_special_gammaln_negative_branch():
    # The reflection formula covers x < 0, including the fractional band
    # between consecutive poles.
    x = torch.tensor([-0.1, -0.5, -1.5, -2.5, -5.5, -10.5, 0.5, 1.0, 2.0, 100.0, 1e30])

    ref = torch.special.gammaln(x)
    tri = special_gammaln(x)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_special_gammaln_poles_and_specials():
    # ATen contract probed on torch CPU fp32: poles (0, negative integers)
    # give +inf, both infinities give +inf, nan stays nan.
    x = torch.tensor([0.0, -0.0, -1.0, -2.0, float("inf"), float("-inf"), float("nan")])

    ref = torch.special.gammaln(x)
    tri = special_gammaln(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == float("inf") and tri[2].item() == float("inf")


def test_special_gammaln_out_variant():
    x = torch.rand(256, dtype=torch.float32) * 10.0 + 0.1
    out = torch.empty_like(x)

    res = special_gammaln_out(x, out=out)
    assert res is out
    torch.testing.assert_close(out, torch.special.gammaln(x), rtol=1e-3, atol=1e-4)


def test_special_gammaln_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_gammaln")
    launches = []
    real_run = mod.special_gammaln_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_gammaln_func, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 20.0 + 0.1
    out = mod.special_gammaln(x)
    assert len(launches) >= 1, "special_gammaln must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.gammaln(x), rtol=1e-3, atol=1e-4)
