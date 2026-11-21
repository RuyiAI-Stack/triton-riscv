import importlib

import pytest
import torch

from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
    special_shifted_chebyshev_polynomial_u_,
)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_shifted_chebyshev_polynomial_u_scalar_n(n):
    torch.manual_seed(0)
    x = torch.rand(512, dtype=torch.float32) * 2.0 - 0.5

    ref = torch.special.shifted_chebyshev_polynomial_u(x, n)
    tri = special_shifted_chebyshev_polynomial_u(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: second-kind recurrence on the shifted argument in
    # fp32; 1e-5 relative up to degree 10.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_shifted_chebyshev_polynomial_u_tensor_n():
    torch.manual_seed(0)
    x = torch.rand(64, dtype=torch.float32) * 2.0 - 0.5
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.shifted_chebyshev_polynomial_u(x.unsqueeze(1), n)
    tri = special_shifted_chebyshev_polynomial_u(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_shifted_chebyshev_polynomial_u_known_rows():
    # U*_0 = 1, U*_1 = 4x - 2 (probed on torch CPU fp32), n < 0 = 0.
    x = torch.tensor([0.5, -0.5], dtype=torch.float32)

    assert special_shifted_chebyshev_polynomial_u(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(
        special_shifted_chebyshev_polynomial_u(x, 1), 4.0 * x - 2.0, rtol=0, atol=0
    )
    torch.testing.assert_close(
        special_shifted_chebyshev_polynomial_u(x, -1),
        torch.zeros_like(x),
        rtol=0,
        atol=0,
    )


def test_special_shifted_chebyshev_polynomial_u_inplace():
    x = torch.rand(256, dtype=torch.float32) * 2.0 - 0.5
    ref = torch.special.shifted_chebyshev_polynomial_u(x, 3)

    tri = x.clone()
    out = special_shifted_chebyshev_polynomial_u_(tri, 3)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_shifted_chebyshev_polynomial_u_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_shifted_chebyshev_polynomial_u")
    launches = []
    real_run = mod.shifted_chebyshev_polynomial_u_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.shifted_chebyshev_polynomial_u_kernel, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 2.0 - 0.5
    out = mod.special_shifted_chebyshev_polynomial_u(x, 4)
    assert len(launches) >= 1, "shifted chebyshev_u must launch a kernel"
    torch.testing.assert_close(
        out,
        torch.special.shifted_chebyshev_polynomial_u(x, 4),
        rtol=1e-5,
        atol=1e-5,
    )
