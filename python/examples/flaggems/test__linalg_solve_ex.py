import torch

from ._linalg_solve_ex import _linalg_solve_ex
from ._refdet import spd


def test__linalg_solve_ex():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    got = _linalg_solve_ex(a, b)
    torch.testing.assert_close(a @ got.result, b, rtol=1e-4, atol=1e-4)
