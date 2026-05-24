import pytest
import torch

from .hstack import hstack


@pytest.mark.parametrize("n_tensors", [2, 3])
@pytest.mark.parametrize("size", [4, 512, 1023, 1024])
def test_hstack_1d(n_tensors, size):
    torch.manual_seed(0)
    tensors = [
        torch.randn(size, dtype=torch.float32, device="cpu")
        for _ in range(n_tensors)
    ]
    ref = torch.hstack(tensors)
    tri = hstack(tensors)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_tensors", [2, 3])
@pytest.mark.parametrize("shape", [(4, 8), (512, 64), (1023, 64), (1024, 64)])
def test_hstack_2d(n_tensors, shape):
    torch.manual_seed(0)
    tensors = [
        torch.randn(shape, dtype=torch.float32, device="cpu")
        for _ in range(n_tensors)
    ]
    ref = torch.hstack(tensors)
    tri = hstack(tensors)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_hstack_varying_shapes_1d():
    torch.manual_seed(0)
    a = torch.randn(3, dtype=torch.float32, device="cpu")
    b = torch.randn(5, dtype=torch.float32, device="cpu")
    c = torch.randn(2, dtype=torch.float32, device="cpu")
    ref = torch.hstack([a, b, c])
    tri = hstack([a, b, c])
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_hstack_varying_shapes_2d():
    torch.manual_seed(0)
    a = torch.randn(4, 3, dtype=torch.float32, device="cpu")
    b = torch.randn(4, 5, dtype=torch.float32, device="cpu")
    c = torch.randn(4, 2, dtype=torch.float32, device="cpu")
    ref = torch.hstack([a, b, c])
    tri = hstack([a, b, c])
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_hstack_single_tensor():
    a = torch.randn(4, 4, dtype=torch.float32, device="cpu")
    ref = torch.hstack([a])
    tri = hstack([a])
    torch.testing.assert_close(tri, ref)
