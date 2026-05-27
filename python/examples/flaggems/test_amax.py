import pytest
import torch

from .amax import amax


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_amax_flattened(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.amax(x)
    tri_out = amax(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_amax_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.amax(x, dim=dim, keepdim=keepdim)
    tri_out = amax(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 8, 16)])
@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (-1, -2)])
@pytest.mark.parametrize("keepdim", [True, False])
def test_amax_dims(shape, dims, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.amax(x, dim=dims, keepdim=keepdim)
    tri_out = amax(x, dim=dims, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
