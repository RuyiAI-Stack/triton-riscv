import torch

from .linalg_matrix_norm import linalg_matrix_norm


def test_linalg_matrix_norm():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    torch.testing.assert_close(
        linalg_matrix_norm(a), torch.linalg.matrix_norm(a), rtol=1e-4, atol=1e-4
    )
