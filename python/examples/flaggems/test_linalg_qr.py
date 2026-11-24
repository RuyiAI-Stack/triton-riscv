import torch

from .linalg_qr import linalg_qr


def test_linalg_qr():
    torch.manual_seed(0)
    source = torch.randn(4, 3)
    q, r = linalg_qr(source)
    torch.testing.assert_close(q @ r, source, rtol=1e-4, atol=1e-4)
