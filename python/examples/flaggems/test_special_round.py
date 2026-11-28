import importlib

import pytest
import torch

from .special_round import special_round, special_round_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_round_half_even(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 100.0

    ref = torch.special.round(x)
    tri = special_round(x)

    # Half-to-even is exact (the kernel does it with floor arithmetic, not
    # libdevice.rint which does not lower on this toolchain), so the
    # comparison is bit-exact.
    assert tri.dtype == ref.dtype
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_special_round_boundary_halves():
    # ATen rounding contract probed on torch CPU fp32: exact halves go to
    # the even neighbour, including signed zeros for the ±0.5 rows.
    x = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 0.4999999, 0.5000001])

    ref = torch.special.round(x)
    tri = special_round(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert tri.tolist() == [0.0, 2.0, 2.0, 4.0, -0.0, -2.0, -2.0, -4.0, 0.0, 1.0]


def test_special_round_special_values():
    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, -0.0])

    ref = torch.special.round(x)
    tri = special_round(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_round_decimals():
    # aten::round.decimals exists upstream; the port forwards the same
    # parameter (`decimals=2` probed on torch CPU: 1.234 -> 1.23).
    x = torch.tensor([1.234, 2.567, -1.235, 0.005], dtype=torch.float32)

    ref = torch.special.round(x, decimals=2)
    tri = special_round(x, decimals=2)

    # the scale-and-round arithmetic is exact in fp32, but torch may fold
    # the 10^decimals divisor in double precision, so allow one ulp.
    torch.testing.assert_close(tri, ref, rtol=0, atol=1e-6)


def test_special_round_out_variant():
    x = torch.randn(256, dtype=torch.float32) * 10.0
    out = torch.empty_like(x)

    res = special_round_out(x, out)
    assert res is out
    torch.testing.assert_close(out, torch.special.round(x), rtol=0, atol=0)


def test_special_round_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_round")
    launches = []
    real_run = mod._special_round_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._special_round_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.special_round(x)
    assert len(launches) >= 1, "special_round must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.round(x), rtol=0, atol=0)
