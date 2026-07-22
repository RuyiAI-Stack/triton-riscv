from math import prod

import pytest
import torch

from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v


@pytest.mark.parametrize("shape", [(512,), (16, 257)])
@pytest.mark.parametrize("n", [0, 3, 9])
def test_special_chebyshev_polynomial_v(shape, n):
    x = torch.linspace(-0.8, 0.8, prod(shape), device="cpu")
    x = x.reshape(shape)

    out = special_chebyshev_polynomial_v(x, n)
    ref = torch.special.chebyshev_polynomial_v(x, n)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_special_chebyshev_polynomial_v_scalar_degree_matches_torch():
    x = torch.linspace(-0.75, 0.75, 1023, dtype=torch.float32, device="cpu")

    out = special_chebyshev_polynomial_v(x, 7)
    ref = torch.special.chebyshev_polynomial_v(x, 7)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [torch.tensor(2), torch.tensor([[0, 1, 2, 7]])])
def test_chebyshev_v_broadcast_and_real_domain(dtype, n):
    x = torch.tensor([[-2.0], [-1.0], [-0.5], [1.0], [2.0]], dtype=dtype)
    actual = special_chebyshev_polynomial_v(x, n)
    expected = torch.special.chebyshev_polynomial_v(x, n)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("n", [-2, -0.5, 0.5, 2.9])
def test_chebyshev_v_degree_conversion(n):
    x = torch.tensor([-2.0, -1.0, 0.5, 2.0])
    torch.testing.assert_close(
        special_chebyshev_polynomial_v(x, n), torch.special.chebyshev_polynomial_v(x, n)
    )
