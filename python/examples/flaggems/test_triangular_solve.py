import torch

from ._refdet import spd
from .cholesky import cholesky
from .triangular_solve import triangular_solve


def test_triangular_solve():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    lower = cholesky(a)
    got, _cloned = triangular_solve(b, lower, upper=False)
    torch.testing.assert_close(lower @ got, b, rtol=1e-4, atol=1e-4)
