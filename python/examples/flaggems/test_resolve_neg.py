import torch

from .resolve_neg import resolve_neg


def test_resolve_neg():
    x = torch.randn(512, dtype=torch.float32, device="cpu")
    tri = resolve_neg(x)
    ref = torch.resolve_neg(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
