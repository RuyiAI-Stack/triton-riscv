import importlib

import pytest
import torch

from .special_bessel_j0 import special_bessel_j0


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_bessel_j0(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 8.0

    ref = torch.special.bessel_j0(x)
    tri = special_bessel_j0(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: j0 is the Cephes rational approximation below 8
    # and the asymptotic expansion above, evaluated in fp32; 1e-5 relative
    # holds on both branches and the atol covers the zeros of j0.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_bessel_j0_branches():
    # The rational/asymptotic split and the zeros at 2.4048, 5.5201,
    # 8.6537 fall inside this sweep.
    x = torch.cat(
        [
            torch.linspace(-30.0, 30.0, 601),
            torch.tensor([0.0, 2.404826, 5.520078, 8.653728, 30.0, 60.0, 120.0]),
        ]
    )

    ref = torch.special.bessel_j0(x)
    tri = special_bessel_j0(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_bessel_j0_special_values():
    # ATen contract probed on torch CPU fp32: j0(0) = 1, j0 is even, and
    # non-finite input (either sign) is nan.
    x = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")], dtype=torch.float32
    )

    ref = torch.special.bessel_j0(x)
    tri = special_bessel_j0(x)

    assert tri[0].item() == 1.0 and tri[1].item() == 1.0
    assert bool(tri[2:].isnan().all()), "j0 at non-finite input is nan on torch CPU"
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_bessel_j0_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_bessel_j0")
    launches = []
    real_run = mod._j0_flat_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._j0_flat_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.special_bessel_j0(x)
    assert len(launches) >= 1, "special_bessel_j0 must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.bessel_j0(x), rtol=1e-5, atol=1e-6)
