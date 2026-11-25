import torch

from .linalg_cross import linalg_cross


def test_linalg_cross():
    torch.manual_seed(0)
    lhs = torch.randn(2, 3)
    rhs = torch.randn(2, 3)
    torch.testing.assert_close(
        linalg_cross(lhs, rhs), torch.linalg.cross(lhs, rhs), rtol=1e-4, atol=1e-4
    )
