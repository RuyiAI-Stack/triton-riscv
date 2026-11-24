import torch

from .linalg_matrix_exp import linalg_matrix_exp


def test_linalg_matrix_exp():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    small = a * 0.1
    torch.testing.assert_close(
        linalg_matrix_exp(small), torch.linalg.matrix_exp(small), rtol=1e-4, atol=1e-4
    )
