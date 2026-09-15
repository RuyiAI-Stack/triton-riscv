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
