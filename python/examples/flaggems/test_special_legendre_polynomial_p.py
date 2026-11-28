import importlib

import pytest
import torch

from .special_legendre_polynomial_p import special_legendre_polynomial_p


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_legendre_polynomial_p_scalar_n(n):
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 1.5

    ref = torch.special.legendre_polynomial_p(x, n)
    tri = special_legendre_polynomial_p(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: Legendre recurrence (k P_k = (2k-1) x P_{k-1} -
    # (k-1) P_{k-2}) in fp32; 1e-5 relative up to degree 10 including
    # |x| > 1 extrapolation.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_legendre_polynomial_p_tensor_n():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float32) * 1.5
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.legendre_polynomial_p(x.unsqueeze(1), n)
    tri = special_legendre_polynomial_p(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_legendre_polynomial_p_known_rows():
    # P_0 = 1, P_1 = x, P_2 = (3x^2 - 1)/2, P_3 = (5x^3 - 3x)/2 (probed on
    # torch CPU fp32: P_3(2) = 17), n < 0 = 0.
    x = torch.tensor([0.5, 2.0], dtype=torch.float32)

    assert special_legendre_polynomial_p(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(special_legendre_polynomial_p(x, 1), x, rtol=0, atol=0)
    torch.testing.assert_close(
        special_legendre_polynomial_p(x, 2),
        torch.tensor([-0.125, 5.5]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        special_legendre_polynomial_p(x, 3),
        torch.tensor([-0.4375, 17.0]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        special_legendre_polynomial_p(x, -1), torch.zeros_like(x), rtol=0, atol=0
    )


def test_special_legendre_polynomial_p_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_legendre_polynomial_p")
    launches = []
    real_run = mod.special_legendre_polynomial_p_forward.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_legendre_polynomial_p_forward, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    out = mod.special_legendre_polynomial_p(x, 4)
    assert len(launches) >= 1, "special_legendre_polynomial_p must launch a kernel"
    torch.testing.assert_close(
        out, torch.special.legendre_polynomial_p(x, 4), rtol=1e-5, atol=1e-5
    )
