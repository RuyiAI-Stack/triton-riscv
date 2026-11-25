import torch

from ._refdet import spd
from .cholesky import cholesky


def test_cholesky():
    torch.manual_seed(0)
    a = spd()
    lower = cholesky(a)
    torch.testing.assert_close(lower @ lower.T, a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        torch.triu(lower, diagonal=1), torch.zeros(3, 3), rtol=0, atol=1e-4
    )
