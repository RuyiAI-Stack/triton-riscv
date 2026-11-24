import torch

from ._refdet import spd
from .linalg_solve import linalg_solve


def test_linalg_solve():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    torch.testing.assert_close(a @ linalg_solve(a, b), b, rtol=1e-4, atol=1e-4)
