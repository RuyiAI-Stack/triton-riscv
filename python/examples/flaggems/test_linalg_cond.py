import torch

from ._refdet import spd
from .linalg_cond import linalg_cond
from .linalg_eigvalsh import linalg_eigvalsh


def test_linalg_cond():
    torch.manual_seed(0)
    a = spd()
    eig = linalg_eigvalsh(a)
    torch.testing.assert_close(
        linalg_cond(a), eig.max() / eig.min(), rtol=1e-4, atol=1e-4
    )
