import torch

from ._refdet import spd
from .linalg_inv import linalg_inv


def test_linalg_inv():
    torch.manual_seed(0)
    a = spd()
    torch.testing.assert_close(linalg_inv(a) @ a, torch.eye(3), rtol=1e-4, atol=1e-4)
