import torch

from ._refdet import spd
from .cholesky import cholesky
from .cholesky_solve import cholesky_solve


def test_cholesky_solve():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    got = cholesky_solve(b, cholesky(a))
    torch.testing.assert_close(a @ got, b, rtol=1e-4, atol=1e-4)
