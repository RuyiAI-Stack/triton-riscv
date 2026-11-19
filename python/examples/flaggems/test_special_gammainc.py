import importlib

import pytest
import torch

from .special_gammainc import special_gammainc


@pytest.mark.parametrize("size", [512, 1023])
def test_special_gammainc(size):
    torch.manual_seed(0)
    a = torch.rand(size, dtype=torch.float32) * 9.0 + 0.5
    x = torch.rand(size, dtype=torch.float32) * 20.0

    ref = torch.special.gammainc(a, x)
    tri = special_gammainc(a, x)

    assert tri.dtype == ref.dtype
    # Tolerance source: the regularized lower incomplete gamma splits into
    # a series (x < a + 1) and a continued fraction above; 1e-5 relative
    # covers the fp32 evaluation on both branches.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-7)


def test_special_gammainc_shape_matrix():
    # a fixed, x varying and both varying: the branch switch at x = a + 1
    # must be seamless.
    a = torch.tensor([1.0, 2.0, 0.5, 5.0, 10.0]).repeat_interleave(41)
    x = torch.arange(0.0, 20.5, 0.5).repeat(5)

    ref = torch.special.gammainc(a, x)
    tri = special_gammainc(a, x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-6)


def test_special_gammainc_boundaries_and_specials():
    # ATen contract probed on torch CPU fp32: gammainc(a, inf) = 1,
    # gammainc(a, 0) = 0, gammainc(a, nan) = nan, a < 0 = nan.
    a = torch.tensor([1.0, 1.0, 1.0, -1.0, float("nan")])
    x = torch.tensor([float("inf"), 0.0, float("nan"), 1.0, 1.0])

    ref = torch.special.gammainc(a, x)
    tri = special_gammainc(a, x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 1.0 and tri[1].item() == 0.0
    assert bool(torch.isnan(tri[3]))


def test_special_gammainc_out_variant():
    a = torch.rand(256, dtype=torch.float32) * 4.0 + 0.5
    x = torch.rand(256, dtype=torch.float32) * 10.0
    out = torch.empty_like(a)

    res = special_gammainc(a, x, out=out)
    assert res is out
    torch.testing.assert_close(out, torch.special.gammainc(a, x), rtol=1e-5, atol=1e-7)


def test_special_gammainc_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_gammainc")
    launches = []
    real_run = mod.special_gammainc_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_gammainc_func, "run", spy_run)
    a = torch.rand(1024, dtype=torch.float32) * 4.0 + 0.5
    x = torch.rand(1024, dtype=torch.float32) * 10.0
    out = mod.special_gammainc(a, x)
    assert len(launches) >= 1, "special_gammainc must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.gammainc(a, x), rtol=1e-5, atol=1e-7)
