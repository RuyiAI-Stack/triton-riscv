import torch

from .linalg_lu_factor import linalg_lu_factor
from .linalg_lu_solve import linalg_lu_solve


def test_linalg_lu_solve():
    torch.manual_seed(0)
    g = torch.randn(3, 3)
    b = torch.randn(3, 2)
    lu, pivots = linalg_lu_factor(g)
    got = linalg_lu_solve(lu, pivots, b)
    torch.testing.assert_close(g @ got, b, rtol=1e-4, atol=1e-4)
