import torch

from .linalg__powsum import linalg__powsum


def test_linalg__powsum():
    torch.manual_seed(0)
    x = torch.randn(6)
    torch.testing.assert_close(
        linalg__powsum(x, 2), x.square().sum(), rtol=1e-4, atol=1e-4
    )
