import torch

from .linalg_tensorinv import linalg_tensorinv


def test_linalg_tensorinv():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    inv = linalg_tensorinv(a.reshape(1, 3, 1, 3), ind=2)
    torch.testing.assert_close(
        inv.reshape(3, 3) @ a, torch.eye(3), rtol=1e-4, atol=1e-4
    )
