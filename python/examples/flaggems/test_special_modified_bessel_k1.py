import importlib

import pytest
import torch

from .special_modified_bessel_k1 import (
    special_modified_bessel_k1,
    special_modified_bessel_k1_out,
)


@pytest.mark.parametrize("size", [512, 1023])
def test_special_modified_bessel_k1(size):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=torch.float32) * 20.0 + 1e-3

    ref = torch.special.modified_bessel_k1(x)
    tri = special_modified_bessel_k1(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: k1 combines the Cephes rational with the 1/x term
    # below 2 and the exp-scaled asymptotic form above, fp32 throughout;
    # 1e-5 relative with a 1e-6 atol away from the near-zero blowup.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_modified_bessel_k1_branches():
    # Both branches plus the 1/sqrt(x) growth toward zero.
    x = torch.cat(
        [
            torch.linspace(1e-4, 2.0, 201),
            torch.linspace(2.0, 30.0, 281),
            torch.tensor([1e-6, 1e-3, 2.0, 30.0, 80.0]),
        ]
    ).to(torch.float32)

    ref = torch.special.modified_bessel_k1(x)
    tri = special_modified_bessel_k1(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_modified_bessel_k1_domain_edges():
    # ATen contract probed on torch CPU fp32: k1(0) = +inf, k1(x < 0) =
    # nan, k1(inf) = 0.
    x = torch.tensor([0.0, -1.0, float("inf"), float("nan")], dtype=torch.float32)

    ref = torch.special.modified_bessel_k1(x)
    tri = special_modified_bessel_k1(x)

    assert tri[0].item() == float("inf") and tri[2].item() == 0.0
    assert bool(tri[1].isnan())
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_modified_bessel_k1_out_variant():
    x = torch.rand(256, dtype=torch.float32) * 5.0 + 0.1
    out = torch.empty_like(x)

    res = special_modified_bessel_k1_out(x, out)
    assert res is out
    torch.testing.assert_close(
        out, torch.special.modified_bessel_k1(x), rtol=1e-5, atol=1e-6
    )


def test_special_modified_bessel_k1_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_modified_bessel_k1")
    launches = []
    real_run = mod.special_modified_bessel_k1_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_modified_bessel_k1_func, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 10.0 + 0.1
    out = mod.special_modified_bessel_k1(x)
    assert len(launches) >= 1, "special_modified_bessel_k1 must launch the kernel"
    torch.testing.assert_close(
        out, torch.special.modified_bessel_k1(x), rtol=1e-5, atol=1e-6
    )
