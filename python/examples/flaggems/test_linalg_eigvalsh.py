import torch

from ._refdet import spd
from .linalg_eigh import linalg_eigh
from .linalg_eigvalsh import linalg_eigvalsh


def test_linalg_eigvalsh():
    torch.manual_seed(0)
    a = spd()
    torch.testing.assert_close(
        linalg_eigvalsh(a), linalg_eigh(a).eigenvalues, rtol=1e-4, atol=1e-4
    )
