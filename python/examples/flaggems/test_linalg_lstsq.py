import torch

from ._refdet import spd
from .linalg_lstsq import linalg_lstsq


def test_linalg_lstsq():
    torch.manual_seed(0)
    a = spd()
    b = torch.randn(3, 2)
    got = linalg_lstsq(a, b, rcond=1e-5).solution
    torch.testing.assert_close(a @ got, b, rtol=1e-4, atol=1e-4)
