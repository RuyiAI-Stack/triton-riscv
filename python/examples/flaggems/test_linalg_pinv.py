import torch

from ._refdet import spd
from .linalg_pinv import linalg_pinv


def test_linalg_pinv():
    torch.manual_seed(0)
    a = spd()
    torch.testing.assert_close(linalg_pinv(a) @ a, torch.eye(3), rtol=1e-4, atol=1e-4)
