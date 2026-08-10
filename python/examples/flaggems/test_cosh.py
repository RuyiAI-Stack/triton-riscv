import pytest
import torch

from .cosh import cosh, cosh_, cosh_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_cosh(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.cosh(x)
    tri = cosh(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_cosh_inplace():
    x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    torch.cosh(x_ref, out=x_ref)
    cosh_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_cosh_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.cosh(x)
    out = torch.empty(shape, dtype=torch.float32, device="cpu")
    tri = cosh_out(x, out=out)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
