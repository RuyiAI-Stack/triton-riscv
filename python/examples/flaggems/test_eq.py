import pytest
import torch

from .eq import eq, eq_scalar, equal


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_eq(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = x == y
    tri = eq(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_eq_scalar():
    x = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32, device="cpu")
    tri = eq(x, 0.0)
    ref = x == 0.0
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_eq_scalar_func(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    scalar_val = 0.5
    ref = x == scalar_val
    tri = eq_scalar(x, scalar_val)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_equal():
    x = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32, device="cpu")
    y = x.clone()
    assert equal(x, y)

    y[0] = 99.0
    assert not equal(x, y)

    z = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cpu")
    assert not equal(x, z)

    scalar_tensor = torch.tensor(42.0, device="cpu")
    assert equal(scalar_tensor, scalar_tensor.clone())
