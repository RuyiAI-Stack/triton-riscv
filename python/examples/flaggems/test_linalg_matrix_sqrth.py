import torch

from .linalg_matrix_sqrth import linalg_matrix_sqrth


def test_linalg_matrix_sqrth():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    root = linalg_matrix_sqrth(a)
    torch.testing.assert_close(root @ root, a, rtol=1e-4, atol=1e-4)
