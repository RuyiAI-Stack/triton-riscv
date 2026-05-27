import pytest
import torch

from .ge import ge, ge_scalar


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_ge(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = x >= y
    tri = ge(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_ge_broadcast(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape[0], 1, dtype=torch.float32, device="cpu")

    ref = x >= y
    tri = ge(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_ge_scalar():
    x = torch.tensor([0.0, 1.0, 2.0, -1.0], dtype=torch.float32, device="cpu")
    tri = ge(x, 1.0)
    ref = x >= 1.0
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_ge_scalar_func(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    scalar_val = 0.0
    ref = x >= scalar_val
    tri = ge_scalar(x, scalar_val)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_ge_int(shape):
    torch.manual_seed(0)
    x = torch.randint(-10, 10, shape, dtype=torch.int32, device="cpu")
    y = torch.randint(-10, 10, shape, dtype=torch.int32, device="cpu")

    ref = x >= y
    tri = ge(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
