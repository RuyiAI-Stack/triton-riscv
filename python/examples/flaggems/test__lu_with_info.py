import torch

from ._lu_with_info import _lu_with_info
from .lu_unpack import lu_unpack


def test__lu_with_info():
    torch.manual_seed(0)
    g = torch.randn(3, 3)
    got = _lu_with_info(g)
    perm, lower, upper = lu_unpack(got.LU, got.pivots)
    torch.testing.assert_close(perm @ lower @ upper, g, rtol=1e-4, atol=1e-4)
    assert int(got.info) == 0
