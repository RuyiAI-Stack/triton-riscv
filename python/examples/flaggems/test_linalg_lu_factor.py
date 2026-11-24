import torch

from .linalg_lu_factor import linalg_lu_factor
from .lu_unpack import lu_unpack


def test_linalg_lu_factor():
    torch.manual_seed(0)
    g = torch.randn(3, 3)
    lu, pivots = linalg_lu_factor(g)
    perm, lower, upper = lu_unpack(lu, pivots)
    torch.testing.assert_close(perm @ lower @ upper, g, rtol=1e-4, atol=1e-4)
