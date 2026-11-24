import torch

from ._refdet import spd
from .linalg_inv_ex import linalg_inv_ex


def test_linalg_inv_ex():
    torch.manual_seed(0)
    a = spd()
    got = linalg_inv_ex(a)
    torch.testing.assert_close(got.inverse @ a, torch.eye(3), rtol=1e-4, atol=1e-4)
    assert int(got.info) == 0
