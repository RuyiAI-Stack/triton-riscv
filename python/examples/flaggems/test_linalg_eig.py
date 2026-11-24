import torch

from .linalg_eig import linalg_eig


def test_linalg_eig():
    diag = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    eig, _vecs = linalg_eig(diag)
    torch.testing.assert_close(
        eig.real, torch.tensor([1.0, 2.0, 3.0]), rtol=1e-4, atol=1e-4
    )
