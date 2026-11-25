import torch

from .linalg_lu_factor_ex import linalg_lu_factor_ex
from .lu_unpack import lu_unpack


def test_linalg_lu_factor_ex():
    torch.manual_seed(0)
    g = torch.randn(3, 3)
    got = linalg_lu_factor_ex(g)
    perm, lower, upper = lu_unpack(got.LU, got.pivots)
    torch.testing.assert_close(perm @ lower @ upper, g, rtol=1e-4, atol=1e-4)
    assert int(got.info) == 0
