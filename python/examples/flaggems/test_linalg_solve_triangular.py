import torch

from ._refdet import spd
from .cholesky import cholesky
from .linalg_solve_triangular import linalg_solve_triangular


def test_linalg_solve_triangular():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    lower = cholesky(a)
    got = linalg_solve_triangular(lower, b, upper=False)
    torch.testing.assert_close(lower @ got, b, rtol=1e-4, atol=1e-4)
