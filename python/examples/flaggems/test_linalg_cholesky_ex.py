import torch

from ._refdet import spd
from .linalg_cholesky_ex import linalg_cholesky_ex


def test_linalg_cholesky_ex():
    torch.manual_seed(0)
    a = spd()
    got = linalg_cholesky_ex(a)
    torch.testing.assert_close(got.L @ got.L.T, a, rtol=1e-4, atol=1e-4)
    assert int(got.info) == 0
