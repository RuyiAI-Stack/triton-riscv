import importlib

import pytest
import torch

from .special_psi import special_psi, special_psi_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_psi(size):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=torch.float32) * 30.0 + 0.01

    ref = torch.special.psi(x)
    tri = special_psi(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: digamma is a recurrence + asymptotic series in
    # fp32; 1e-5 relative holds for the shifted argument, and the atol
    # covers the zero crossings of psi around x ~ 1.46.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_psi_recurrence_identity():
    # psi(x + 1) - psi(x) = 1/x pins the recurrence step without pinning
    # the asymptotic coefficients.
    x = torch.tensor([0.25, 1.0, 2.5, 10.0, 100.0], dtype=torch.float32)

    delta = special_psi(x + 1.0) - special_psi(x)
    torch.testing.assert_close(delta, 1.0 / x, rtol=1e-4, atol=1e-6)


def test_special_psi_negative_branch():
    x = torch.tensor([-0.1, -0.5, -1.5, -2.5, -10.5], dtype=torch.float32)

    ref = torch.special.psi(x)
    tri = special_psi(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_psi_special_values():
    # ATen contract probed on torch CPU fp32: psi(0) = -inf, psi(1) = -gamma,
    # psi at negative integers = nan, psi(+inf) = +inf, psi(-inf) = nan.
    x = torch.tensor([0.0, 1.0, -1.0, -2.0, float("inf"), float("-inf"), float("nan")])

    ref = torch.special.psi(x)
    tri = special_psi(x)

    # psi(1) = -gamma is a transcendental value: both sides compute it
    # (torch through calc_digamma, the port through its own recurrence),
    # so the finite rows compare at ulp distance and the sentinels stay
    # exact in the explicit asserts below.
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert tri[0].item() == float("-inf")
    assert abs(tri[1].item() + 0.5772157) < 1e-6
    assert bool(tri[2:4].isnan().all())


def test_special_psi_out_variant():
    x = torch.rand(256, dtype=torch.float32) * 10.0 + 0.1
    out = torch.empty_like(x)

    res = special_psi_out(x, out)
    assert res is out
    torch.testing.assert_close(out, torch.special.psi(x), rtol=1e-5, atol=1e-6)


def test_special_psi_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_psi")
    launches = []
    real_run = mod._psi_flat32.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._psi_flat32, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 15.0 + 0.1
    out = mod.special_psi(x)
    assert len(launches) >= 1, "special_psi must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.psi(x), rtol=1e-5, atol=1e-6)
