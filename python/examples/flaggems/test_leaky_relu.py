import pytest
import torch

from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_leaky_relu(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.nn.functional.leaky_relu(x)
    tri = leaky_relu(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_leaky_relu_inplace():
    x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    torch.nn.functional.leaky_relu(x_ref, inplace=True)
    leaky_relu_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_leaky_relu_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)

    ref = torch.nn.functional.leaky_relu(x)
    leaky_relu_out(x, out=out)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
