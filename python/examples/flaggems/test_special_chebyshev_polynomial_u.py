import importlib

import pytest
import torch

from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_chebyshev_polynomial_u_scalar_n(n):
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 1.5

    ref = torch.special.chebyshev_polynomial_u(x, n)
    tri = special_chebyshev_polynomial_u(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: U_k = 2x U_{k-1} - U_{k-2} on fp32; 1e-5 relative
    # covers degrees up to 10 including the |x| > 1 growth.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_chebyshev_polynomial_u_tensor_n():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float32) * 1.5
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.chebyshev_polynomial_u(x.unsqueeze(1), n)
    tri = special_chebyshev_polynomial_u(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_chebyshev_polynomial_u_known_rows():
    # U_0 = 1, U_1 = 2x (probed on torch CPU fp32), n < 0 = 0.
    x = torch.tensor([0.5, 2.0], dtype=torch.float32)

    assert special_chebyshev_polynomial_u(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(
        special_chebyshev_polynomial_u(x, 1), 2.0 * x, rtol=0, atol=0
    )
    torch.testing.assert_close(
        special_chebyshev_polynomial_u(x, -1), torch.zeros_like(x), rtol=0, atol=0
    )


def test_special_chebyshev_polynomial_u_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_chebyshev_polynomial_u")
    launches = []
    real_run = mod.special_chebyshev_polynomial_u_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_chebyshev_polynomial_u_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    out = mod.special_chebyshev_polynomial_u(x, 4)
    assert len(launches) >= 1, "special_chebyshev_polynomial_u must launch a kernel"
    torch.testing.assert_close(
        out, torch.special.chebyshev_polynomial_u(x, 4), rtol=1e-5, atol=1e-5
    )
