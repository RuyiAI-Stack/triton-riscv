import pytest
import torch

from .cumprod import cumprod, cumprod_


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (1024,)])
@pytest.mark.parametrize("dim", [0, -1])
def test_cumprod(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.cumprod(x, dim=dim)
    tri_out = cumprod(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("dim", [0, 1])
def test_cumprod_inplace(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)
    x_clone = x.clone()
    ref_out = torch.cumprod(x_clone, dim=dim)
    tri_out = cumprod_(x, dim=dim)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 32)])
@pytest.mark.parametrize("dim", [0, 1])
def test_cumprod_int(shape, dim):
    torch.manual_seed(0)
    x = torch.randint(1, 10, shape, device="cpu", dtype=torch.int32)

    ref_out = torch.cumprod(x, dim=dim)
    tri_out = cumprod(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(4, 8, 16)])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_cumprod_3d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.cumprod(x, dim=dim)
    tri_out = cumprod(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
