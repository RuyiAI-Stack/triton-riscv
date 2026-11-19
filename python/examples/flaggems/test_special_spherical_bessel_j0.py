import importlib

import pytest
import torch

from .special_spherical_bessel_j0 import special_spherical_bessel_j0


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_spherical_bessel_j0(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 10.0

    ref = torch.special.spherical_bessel_j0(x)
    tri = special_spherical_bessel_j0(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: sph_j0(x) = sin(x) / x with the x = 0 limit
    # special-cased; fp32 sin and one division, so 1e-6 relative with a
    # 1e-7 atol at the far zeros of sin.
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-7)


def test_special_spherical_bessel_j0_zeros_and_origin():
    # The removable singularity at 0 (value 1) and the zeros of sin(x)
    # where sph_j0 crosses zero with amplitude ~1/x.
    x = torch.tensor(
        [0.0, -0.0, 1e-8, 3.14159265, 6.28318531, -3.14159265, 100.0],
        dtype=torch.float32,
    )

    ref = torch.special.spherical_bessel_j0(x)
    tri = special_spherical_bessel_j0(x)

    assert tri[0].item() == 1.0 and tri[1].item() == 1.0
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-7)


def test_special_spherical_bessel_j0_special_values():
    # ATen contract probed on torch CPU fp32: sph_j0(±inf) = 0,
    # sph_j0(nan) = nan.
    x = torch.tensor([float("inf"), float("-inf"), float("nan")], dtype=torch.float32)

    ref = torch.special.spherical_bessel_j0(x)
    tri = special_spherical_bessel_j0(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 0.0 and tri[1].item() == 0.0


def test_special_spherical_bessel_j0_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_spherical_bessel_j0")
    launches = []
    real_run = mod.spherical_bessel_j0_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.spherical_bessel_j0_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 6.0
    out = mod.special_spherical_bessel_j0(x)
    assert len(launches) >= 1, "special_spherical_bessel_j0 must launch the kernel"
    torch.testing.assert_close(
        out, torch.special.spherical_bessel_j0(x), rtol=1e-6, atol=1e-7
    )
