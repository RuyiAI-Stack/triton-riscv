import math

import pytest
import torch

from .hadamard_transform import hadamard_transform


def _hadamard_matrix(n):
    """Construct a Hadamard matrix of order n (n must be a power of 2)."""
    h = torch.tensor([[1]], dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


@pytest.mark.parametrize("dim", [8, 16, 32, 64, 128, 256, 512])
def test_hadamard_transform_power_of_2(dim):
    """Test hadamard_transform on power-of-2 dims against explicit matrix multiply."""
    torch.manual_seed(0)
    x = torch.randn(2, dim, device="cpu", dtype=torch.float32)
    h = _hadamard_matrix(dim)
    ref = torch.matmul(x, h.T)
    tri = hadamard_transform(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dim", [512, 1023, 1024])
def test_hadamard_transform_scaled(dim):
    """Test hadamard_transform with scaling factor."""
    torch.manual_seed(0)
    x = torch.randn(3, dim, device="cpu", dtype=torch.float32)
    # For non-power-of-2 dims, the function pads to next power of 2, runs FHT, trims back.
    n = 1 << math.ceil(math.log2(dim)) if dim > 1 else 1
    h = _hadamard_matrix(n)
    x_padded = torch.zeros(3, n, dtype=torch.float32)
    x_padded[:, :dim] = x
    scale = 0.5
    ref = torch.mul(torch.matmul(x_padded, h.T), scale)
    ref = ref[:, :dim]
    tri = hadamard_transform(x, scale=scale)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_hadamard_transform_autograd():
    """Test hadamard_transform with autograd (backward = forward for Hadamard)."""
    torch.manual_seed(0)
    x = torch.randn(2, 64, device="cpu", dtype=torch.float32, requires_grad=True)
    y = hadamard_transform(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    # Hadamard matrix is self-inverse up to scale: grad = sum(hadamard(ones))
    # Forward + backward of sum is just row sums of Hadamard
    h = _hadamard_matrix(64)
    expected_grad = torch.mul(torch.matmul(torch.ones(2, 64), h.T), 1.0)
    torch.testing.assert_close(x.grad, expected_grad, rtol=1e-4, atol=1e-4)
