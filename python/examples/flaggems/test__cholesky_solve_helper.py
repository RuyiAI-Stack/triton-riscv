import torch

from ._cholesky_solve_helper import _cholesky_solve_helper
from ._refdet import spd
from .cholesky import cholesky


def test__cholesky_solve_helper():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    got = _cholesky_solve_helper(b, cholesky(a), False)
    torch.testing.assert_close(a @ got, b, rtol=1e-4, atol=1e-4)
