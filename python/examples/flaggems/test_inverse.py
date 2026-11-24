import torch

from ._refdet import spd
from .inverse import inverse


def test_inverse():
    torch.manual_seed(0)
    a = spd()
    torch.testing.assert_close(inverse(a) @ a, torch.eye(3), rtol=1e-4, atol=1e-4)
