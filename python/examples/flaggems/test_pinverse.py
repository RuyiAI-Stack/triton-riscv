import torch

from ._refdet import spd
from .pinverse import pinverse


def test_pinverse():
    torch.manual_seed(0)
    a = spd()
    torch.testing.assert_close(pinverse(a) @ a, torch.eye(3), rtol=1e-4, atol=1e-4)
