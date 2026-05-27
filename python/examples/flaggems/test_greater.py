import pytest
import torch

from .greater import greater, greater_out, greater_scalar, greater_scalar_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_greater(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.greater(x, y)
    tri = greater(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_greater_scalar():
    x = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32, device="cpu")
    tri = greater_scalar(x, 0.0)
    ref = torch.greater(x, 0.0)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_greater_out():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    y = torch.randn(1024, dtype=torch.float32, device="cpu")
    ref = torch.greater(x, y)

    out = torch.empty(1024, dtype=torch.bool, device="cpu")
    tri = greater_out(x, y, out=out)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_greater_scalar_out():
    x = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32, device="cpu")
    ref = torch.greater(x, 0.0)

    out = torch.empty_like(x, dtype=torch.bool)
    tri = greater_scalar_out(x, 0.0, out=out)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
