import pytest
import torch

from .argmax import argmax


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_argmax_flattened(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.argmax(x)
    tri_out = argmax(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_argmax_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.argmax(x, dim=dim, keepdim=keepdim)
    tri_out = argmax(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(4, 8, 16)])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_argmax_3d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.argmax(x, dim=dim)
    tri_out = argmax(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out)
