import importlib

import pytest
import torch

from .special_ndtri import special_ndtri


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_ndtri(size):
    torch.manual_seed(0)
    x = torch.rand(size, dtype=torch.float32)

    ref = torch.special.ndtri(x)
    tri = special_ndtri(x)

    assert tri.dtype == ref.dtype
    # Tolerance source: ndtri is the inverse normal CDF, evaluated with the
    # aten rational approximation in fp32; the tails amplify input ulp
    # changes, so relative slack of 1e-4 with a small atol keeps both the
    # bulk and the tails meaningful.
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-5)


def test_special_ndtri_round_trip():
    # ndtr(ndtri(p)) == p is the defining property; it pins the branch
    # split (central rational vs tail asymptotics) without hardcoding
    # coefficient values.
    p = torch.tensor(
        [1e-8, 1e-5, 1e-3, 0.01, 0.02, 0.5, 0.98, 0.99, 0.999, 1 - 1e-5, 1 - 1e-8],
        dtype=torch.float32,
    )

    z = special_ndtri(p)
    back = torch.special.ndtr(z)
    torch.testing.assert_close(back, p, rtol=1e-4, atol=1e-7)
    assert z[5].item() == 0.0


def test_special_ndtri_special_values():
    # ATen contract probed on torch CPU fp32: ndtri(0) = -inf,
    # ndtri(1) = +inf, values outside [0, 1] = nan, nan stays nan.
    x = torch.tensor([0.0, 1.0, -0.5, 1.5, float("nan")])

    ref = torch.special.ndtri(x)
    tri = special_ndtri(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == float("-inf") and tri[1].item() == float("inf")


def test_special_ndtri_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_ndtri")
    launches = []
    real_run = mod._ndtri_flat_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._ndtri_flat_kernel, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32)
    out = mod.special_ndtri(x)
    assert len(launches) >= 1, "special_ndtri must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.ndtri(x), rtol=1e-4, atol=1e-5)
