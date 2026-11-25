import torch

from ._refdet import spd
from .cholesky import cholesky
from .cholesky_inverse import cholesky_inverse


def test_cholesky_inverse():
    torch.manual_seed(0)
    a = spd()
    got = cholesky_inverse(cholesky(a))
    torch.testing.assert_close(got @ a, torch.eye(3), rtol=1e-4, atol=1e-4)
