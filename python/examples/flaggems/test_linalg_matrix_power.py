import torch

from .linalg_matrix_power import linalg_matrix_power


def test_linalg_matrix_power():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    torch.testing.assert_close(
        linalg_matrix_power(a, 3), torch.linalg.matrix_power(a, 3), rtol=1e-4, atol=1e-4
    )
