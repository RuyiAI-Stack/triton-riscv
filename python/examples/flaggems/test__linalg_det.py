import torch

from ._linalg_det import _linalg_det
from ._refdet import det3, spd


def test__linalg_det():
    torch.manual_seed(0)
    a = spd()
    torch.testing.assert_close(_linalg_det(a), det3(a), rtol=1e-4, atol=1e-4)
