import torch

from ._refdet import det3, spd
from .linalg_det import linalg_det


def test_linalg_det():
    torch.manual_seed(0)
    a = spd()
    torch.testing.assert_close(linalg_det(a), det3(a), rtol=1e-4, atol=1e-4)
