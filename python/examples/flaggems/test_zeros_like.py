import pytest
import torch

from .zeros_like import zeros_like


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_zeros_like(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.zeros_like(x)
    tri_out = zeros_like(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_zeros_like_dtype(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.zeros_like(x, dtype=torch.float64)
    tri_out = zeros_like(x, dtype=torch.float64)

    assert tri_out.dtype == torch.float64
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 64), (4, 32, 64)])
def test_zeros_like_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.zeros_like(x)
    tri_out = zeros_like(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
