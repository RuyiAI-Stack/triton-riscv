import torch

from ._refdet import spd
from .linalg_solve_ex import linalg_solve_ex


def test_linalg_solve_ex():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    got = linalg_solve_ex(a, b)
    torch.testing.assert_close(a @ got.result, b, rtol=1e-4, atol=1e-4)
    assert int(got.info) == 0
