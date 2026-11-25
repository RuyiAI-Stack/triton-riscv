import torch

from ._refdet import spd
from .linalg_ldl_factor import linalg_ldl_factor
from .linalg_ldl_solve import linalg_ldl_solve


def test_linalg_ldl_solve():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    ld, pivots = linalg_ldl_factor(a)
    got = linalg_ldl_solve(ld, pivots, b)
    torch.testing.assert_close(a @ got, b, rtol=1e-4, atol=1e-4)
