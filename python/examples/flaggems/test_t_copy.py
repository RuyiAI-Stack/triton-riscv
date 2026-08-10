import pytest
import torch

from .t_copy import t_copy, t_copy_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_t_copy_1d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.t(x).contiguous()
    tri = t_copy(x)
    if x.dim() == 1:
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(tri, x.t(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(32, 64), (16, 128)])
def test_t_copy_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.t(x).contiguous()
    tri = t_copy(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_t_copy_out_1d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)
    ref = torch.t(x).contiguous()
    t_copy_out(x, out)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(32, 64), (16, 128)])
def test_t_copy_out_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    M, N = shape
    out = torch.empty((N, M), dtype=torch.float32, device="cpu")
    ref = torch.t(x).contiguous()
    t_copy_out(x, out)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
