import pytest
import torch

from .dot import dot


@pytest.mark.parametrize("n", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dot(n, dtype):
    torch.manual_seed(0)
    x = torch.randn(n, dtype=dtype, device="cpu")
    y = torch.randn(n, dtype=dtype, device="cpu")

    ref = torch.dot(x, y)
    tri = dot(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dot_large(n, dtype):
    torch.manual_seed(0)
    x = torch.randn(n, dtype=dtype, device="cpu")
    y = torch.randn(n, dtype=dtype, device="cpu")

    ref = torch.dot(x, y)
    tri = dot(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
