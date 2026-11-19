import importlib

import pytest
import torch

from .special_bessel_y0 import special_bessel_y0


@pytest.mark.parametrize("size", [512, 1023])
def test_special_bessel_y0(size):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=torch.float32) * 20.0 + 1e-3

    ref = torch.special.bessel_y0(x)
    tri = special_bessel_y0(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: y0 is the Cephes small-argument series (with the
    # j0 companion and the log term) below 8 and the asymptotic expansion
    # above, fp32 throughout; 1e-5 relative with a 1e-6 atol for zeros.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_bessel_y0_branches():
    # Both branches plus the first zeros at 0.8936 and 3.9577.
    x = torch.cat(
        [
            torch.linspace(1e-4, 0.5, 51),
            torch.linspace(0.5, 30.0, 295),
            torch.tensor([0.893577, 3.957678, 7.086051, 30.0, 80.0]),
        ]
    ).to(torch.float32)

    ref = torch.special.bessel_y0(x)
    tri = special_bessel_y0(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_bessel_y0_domain_edges():
    # ATen contract probed on torch CPU fp32: y0(0) = -inf, y0(x < 0) =
    # nan, y0(inf) = nan.
    x = torch.tensor([0.0, -1.0, float("inf"), float("nan")], dtype=torch.float32)

    ref = torch.special.bessel_y0(x)
    tri = special_bessel_y0(x)

    assert tri[0].item() == float("-inf")
    assert bool(tri[1:].isnan().all())
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_bessel_y0_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_bessel_y0")
    launches = []
    real_run = mod._bessel_y0_kernel_contig.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._bessel_y0_kernel_contig, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 10.0 + 0.1
    out = mod.special_bessel_y0(x)
    assert len(launches) >= 1, "special_bessel_y0 must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.bessel_y0(x), rtol=1e-5, atol=1e-6)
