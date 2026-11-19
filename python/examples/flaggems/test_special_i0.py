import importlib

import pytest
import torch

from .special_i0 import special_i0, special_i0_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_i0(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 8.0

    ref = torch.special.i0(x)
    tri = special_i0(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: i0 is the Cephes rational approximation for |x| <=
    # 3.75 and the exp-scaled asymptotic form above, fp32 throughout;
    # 1e-5 relative covers both (the even function doubles nothing).
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_i0_branches():
    # The 3.75 split plus growth to x = 88, where i0 reaches ~1e37.
    x = torch.cat(
        [
            torch.linspace(-20.0, 20.0, 401),
            torch.tensor([3.75, 3.7500001, 20.0, 50.0, 88.0]),
        ]
    )

    ref = torch.special.i0(x)
    tri = special_i0(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)


def test_special_i0_special_values():
    # ATen contract probed on torch CPU fp32: i0 is even, and non-finite
    # input (either sign) is nan on this torch build.
    x = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")], dtype=torch.float32
    )

    ref = torch.special.i0(x)
    tri = special_i0(x)

    assert tri[0].item() == tri[1].item()
    # torch's own i0(0) is 0.99999994 rather than 1.0 (probed), so the
    # +-0 rows differ from our small-branch constant by one ulp; the
    # non-finite rows are exact sentinels.
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_special_i0_out_variant():
    x = torch.randn(256, dtype=torch.float32) * 4.0
    out = torch.empty_like(x)

    res = special_i0_out(x, out)
    assert res is out
    torch.testing.assert_close(out, torch.special.i0(x), rtol=1e-5, atol=1e-6)


def test_special_i0_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_i0")
    launches = []
    real_run = mod._special_i0_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._special_i0_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.special_i0(x)
    assert len(launches) >= 1, "special_i0 must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.i0(x), rtol=1e-5, atol=1e-6)
