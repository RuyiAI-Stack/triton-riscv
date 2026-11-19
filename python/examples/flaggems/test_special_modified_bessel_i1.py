import importlib

import pytest
import torch

from .special_modified_bessel_i1 import (
    special_modified_bessel_i1,
    special_modified_bessel_i1_out,
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_modified_bessel_i1(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 8.0

    ref = torch.special.modified_bessel_i1(x)
    tri = special_modified_bessel_i1(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: modified i1 is the Cephes odd rational below 3.75
    # and the exp-scaled asymptotic form above, fp32 throughout; 1e-5
    # relative with a 1e-6 atol at the zero.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_modified_bessel_i1_odd_and_growth():
    # Odd symmetry plus growth to the fp32 edge of the series branch.
    x = torch.tensor(
        [0.0, 1.0, -1.0, 3.75, -3.75, 20.0, 50.0, 88.0], dtype=torch.float32
    )

    ref = torch.special.modified_bessel_i1(x)
    tri = special_modified_bessel_i1(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tri[2], -tri[1], rtol=1e-6, atol=1e-7)
    assert bool(torch.isfinite(tri).all())


def test_special_modified_bessel_i1_special_values():
    # ATen contract probed on torch CPU fp32: i1(±0) = ±0, non-finite
    # behavior follows torch (pinned by the reference row itself).
    x = torch.tensor([0.0, -0.0, float("inf"), float("nan")], dtype=torch.float32)

    ref = torch.special.modified_bessel_i1(x)
    tri = special_modified_bessel_i1(x)

    assert tri[0].item() == 0.0
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_modified_bessel_i1_out_variant():
    x = torch.randn(256, dtype=torch.float32) * 4.0
    out = torch.empty_like(x)

    res = special_modified_bessel_i1_out(x, out)
    assert res is out
    torch.testing.assert_close(
        out, torch.special.modified_bessel_i1(x), rtol=1e-5, atol=1e-6
    )


def test_special_modified_bessel_i1_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_modified_bessel_i1")
    launches = []
    real_run = mod.special_modified_bessel_i1_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_modified_bessel_i1_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.special_modified_bessel_i1(x)
    assert len(launches) >= 1, "special_modified_bessel_i1 must launch the kernel"
    torch.testing.assert_close(
        out, torch.special.modified_bessel_i1(x), rtol=1e-5, atol=1e-6
    )
