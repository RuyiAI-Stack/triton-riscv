import importlib

import pytest
import torch

from .special_bessel_j1 import special_bessel_j1


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_bessel_j1(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 8.0

    ref = torch.special.bessel_j1(x)
    tri = special_bessel_j1(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: same Cephes rational + asymptotic construction as
    # j0 evaluated in fp32; 1e-5 relative with a 1e-6 atol for the zeros.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_bessel_j1_odd_symmetry_and_zeros():
    # j1 is odd, and its first zeros sit at 0, 3.8317, 7.0156, 10.1735.
    x = torch.tensor(
        [1.0, 3.831706, 7.015587, 10.173468, 25.0, 64.0, -64.0], dtype=torch.float32
    )

    ref = torch.special.bessel_j1(x)
    tri = special_bessel_j1(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tri[6], -tri[5], rtol=1e-6, atol=1e-7)


def test_special_bessel_j1_special_values():
    # ATen contract probed on torch CPU fp32: j1(±0) = ±0, non-finite
    # input is nan.
    x = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")], dtype=torch.float32
    )

    ref = torch.special.bessel_j1(x)
    tri = special_bessel_j1(x)

    assert tri[0].item() == 0.0
    assert bool(tri[2:].isnan().all())
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_bessel_j1_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_bessel_j1")
    launches = []
    real_run = mod.j1_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.j1_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.special_bessel_j1(x)
    assert len(launches) >= 1, "special_bessel_j1 must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.bessel_j1(x), rtol=1e-5, atol=1e-6)
