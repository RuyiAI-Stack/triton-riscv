import torch

from ._refdet import spd
from .linalg_eigh import linalg_eigh


def test_linalg_eigh():
    torch.manual_seed(0)
    a = spd()
    eig, vecs = linalg_eigh(a)
    torch.testing.assert_close(a @ vecs, vecs * eig, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(vecs.T @ vecs, torch.eye(3), rtol=1e-4, atol=1e-4)
