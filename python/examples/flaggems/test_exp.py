import pytest
import torch

from .exp import exp, exp_, exp_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_exp(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    ref = torch.exp(x)
    tri = exp(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_exp_inplace():
    x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone().exp_()
    exp_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_exp_out(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    ref = torch.exp(x)
    out = torch.empty_like(x)
    result = exp_out(x, out)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    assert result is out, "exp_out must return the output tensor"
