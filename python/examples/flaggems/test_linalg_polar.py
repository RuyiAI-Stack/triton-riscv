import torch

from .linalg_polar import linalg_polar


def test_linalg_polar():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    unitary, hermitian = linalg_polar(a)
    torch.testing.assert_close(unitary @ hermitian, a, rtol=1e-4, atol=1e-4)
