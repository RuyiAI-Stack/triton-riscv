import pytest
import torch

from .concatenate import concatenate


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_concatenate_1d(size):
    torch.manual_seed(0)
    a = torch.randn(size, device="cpu", dtype=torch.float32)
    b = torch.randn(size, device="cpu", dtype=torch.float32)
    ref = torch.cat([a, b])
    tri = concatenate([a, b])
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dim", [0, 1])
def test_concatenate_2d(dim):
    torch.manual_seed(0)
    if dim == 0:
        a = torch.randn(4, 8, device="cpu", dtype=torch.float32)
        b = torch.randn(6, 8, device="cpu", dtype=torch.float32)
    else:
        a = torch.randn(4, 8, device="cpu", dtype=torch.float32)
        b = torch.randn(4, 6, device="cpu", dtype=torch.float32)
    ref = torch.cat([a, b], dim=dim)
    tri = concatenate([a, b], dim=dim)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_concatenate_2d_fixed(size):
    torch.manual_seed(0)
    a = torch.randn(4, size, device="cpu", dtype=torch.float32)
    b = torch.randn(6, size, device="cpu", dtype=torch.float32)
    ref = torch.cat([a, b], dim=0)
    tri = concatenate([a, b], dim=0)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_concatenate_three_tensors():
    torch.manual_seed(0)
    a = torch.randn(3, 4, device="cpu", dtype=torch.float32)
    b = torch.randn(5, 4, device="cpu", dtype=torch.float32)
    c = torch.randn(2, 4, device="cpu", dtype=torch.float32)
    ref = torch.cat([a, b, c])
    tri = concatenate([a, b, c])
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
