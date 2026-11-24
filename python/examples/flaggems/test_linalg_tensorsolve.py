import torch

from ._refdet import spd
from .linalg_tensorsolve import linalg_tensorsolve


def test_linalg_tensorsolve():
    torch.manual_seed(0)
    a = spd()
    rhs = torch.randn(3)
    got = linalg_tensorsolve(a, rhs)
    torch.testing.assert_close(a @ got, rhs, rtol=1e-4, atol=1e-4)
