import pytest
import torch

from .ne import ne, ne_scalar


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_ne(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.ne(x, y)
    tri = ne(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_ne_scalar():
    x = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32, device="cpu")
    tri = ne(x, 0.0)
    ref = torch.ne(x, 0.0)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_ne_scalar_func(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    scalar_val = 0.5
    ref = torch.ne(x, scalar_val)
    tri = ne_scalar(x, scalar_val)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
