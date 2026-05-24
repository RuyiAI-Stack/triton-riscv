import pytest
import torch

from .prod import prod, prod_dim


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_prod(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu").abs()

    ref_out = torch.prod(x)
    tri_out = prod(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(32, 16), (64, 8)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_prod_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu").abs()

    ref_out = torch.prod(x, dim=dim, keepdim=keepdim)
    tri_out = prod_dim(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
