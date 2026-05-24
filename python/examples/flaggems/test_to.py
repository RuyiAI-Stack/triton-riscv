import torch

from .to import to_copy


def test_to_copy_same_dtype():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32, device="cpu")
    tri = to_copy(x)
    torch.testing.assert_close(tri, x, rtol=1e-4, atol=1e-4)


def test_to_copy_dtype_change():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32, device="cpu")
    tri = to_copy(x, dtype=torch.float64)
    ref = x.to(dtype=torch.float64)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
