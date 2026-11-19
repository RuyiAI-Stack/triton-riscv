import importlib

import pytest
import torch

from .special_modified_bessel_i0 import (
    special_modified_bessel_i0,
    special_modified_bessel_i0_out,
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_modified_bessel_i0(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 8.0

    ref = torch.special.modified_bessel_i0(x)
    tri = special_modified_bessel_i0(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: modified i0 is the same Cephes rational +
    # exp-scaled asymptotic construction as i0, fp32 throughout; 1e-5
    # relative covers both branches.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_modified_bessel_i0_matches_i0():
    # aten exposes the same function under both names; the port keeps the
    # equality so a later kernel change cannot drift them apart.
    x = torch.tensor([-8.0, -3.0, 0.5, 3.75, 12.0], dtype=torch.float32)

    # aten's i0 differs from the Numerical-Recipes form by 1-2 ulp (its
    # i0(0) is 0.99999994), so the two entry points are pinned at fp32
    # rounding distance, not bitwise.
    torch.testing.assert_close(
        special_modified_bessel_i0(x), torch.special.i0(x), rtol=1e-6, atol=1e-6
    )


def test_special_modified_bessel_i0_special_values():
    # Same probed contract as i0: even at ±0, and non-finite input is nan
    # on this torch build.
    x = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")], dtype=torch.float32
    )

    ref = torch.special.modified_bessel_i0(x)
    tri = special_modified_bessel_i0(x)

    assert tri[0].item() == tri[1].item()
    # Same one-ulp i0(0) difference as the i0 port (probed on torch).
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_special_modified_bessel_i0_out_variant():
    x = torch.randn(256, dtype=torch.float32) * 4.0
    out = torch.empty_like(x)

    res = special_modified_bessel_i0_out(x, out)
    assert res is out
    torch.testing.assert_close(
        out, torch.special.modified_bessel_i0(x), rtol=1e-5, atol=1e-6
    )


def test_special_modified_bessel_i0_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_modified_bessel_i0")
    launches = []
    real_run = mod._special_modified_bessel_i0_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._special_modified_bessel_i0_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.special_modified_bessel_i0(x)
    assert len(launches) >= 1, "special_modified_bessel_i0 must launch the kernel"
    torch.testing.assert_close(
        out, torch.special.modified_bessel_i0(x), rtol=1e-5, atol=1e-6
    )
