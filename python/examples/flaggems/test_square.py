import pytest
import torch

from .square import square, square_, square_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_square(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.square(x)
    tri = square(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_square_inplace():
    x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    torch.square(x_ref, out=x_ref)
    square_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_square_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.square(x)
    out = torch.empty_like(x)
    result = square_out(x, out=out)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    assert result is out, "square_out must return the output tensor"


def test_square_out_none():
    x = torch.randn(128, dtype=torch.float32, device="cpu")
    ref = torch.square(x)
    tri = square_out(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
