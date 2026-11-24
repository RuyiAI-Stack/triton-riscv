import torch

from .linalg_lu import linalg_lu


def test_linalg_lu():
    torch.manual_seed(0)
    g = torch.randn(3, 3)
    perm, lower, upper = linalg_lu(g)
    torch.testing.assert_close(perm @ lower @ upper, g, rtol=1e-4, atol=1e-4)
