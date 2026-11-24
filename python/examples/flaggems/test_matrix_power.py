import torch

from .matrix_power import matrix_power


def test_matrix_power():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    torch.testing.assert_close(
        matrix_power(a, 3), torch.linalg.matrix_power(a, 3), rtol=1e-4, atol=1e-4
    )
