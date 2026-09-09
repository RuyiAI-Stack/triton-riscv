import pytest
import torch

from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
    special_shifted_chebyshev_polynomial_u_,
)


@pytest.mark.parametrize("n", [0, 3, 9])
def test_special_shifted_chebyshev_polynomial_u(n):
    x = torch.linspace(0.0, 1.0, 1023, dtype=torch.float32, device="cpu")

    out = special_shifted_chebyshev_polynomial_u(x, n)
    ref = torch.special.shifted_chebyshev_polynomial_u(x, n)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_special_shifted_chebyshev_polynomial_u_inplace():
    x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32, device="cpu")
    ref = torch.special.shifted_chebyshev_polynomial_u(x, 2)

    ret = special_shifted_chebyshev_polynomial_u_(x, 2)

    assert ret is x
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


def test_special_shifted_chebyshev_polynomial_u_tensor_degree():
    x = torch.linspace(0.1, 0.9, 1023, dtype=torch.float32, device="cpu")
    n = torch.arange(x.numel(), dtype=torch.int64, device="cpu").remainder(10)

    out = special_shifted_chebyshev_polynomial_u(x, n)
    ref = torch.special.shifted_chebyshev_polynomial_u(x, n)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_special_shifted_chebyshev_polynomial_u_scalar_degree_matches_torch():
    x = torch.linspace(0.0, 1.0, 1024, dtype=torch.float32, device="cpu")

    out = special_shifted_chebyshev_polynomial_u(x, 8)
    ref = torch.special.shifted_chebyshev_polynomial_u(x, 8)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
