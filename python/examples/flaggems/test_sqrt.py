import pytest
import torch

from .sqrt import sqrt, sqrt_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_sqrt(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1
    ref = torch.sqrt(x)
    tri = sqrt(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_sqrt_inplace():
    x = torch.tensor([0.25, 1.0, 4.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone().sqrt_()
    sqrt_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
