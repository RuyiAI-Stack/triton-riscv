import pytest
import torch

from .aminmax import aminmax


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_aminmax_flattened(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_min, ref_max = torch.aminmax(x)
    tri_min, tri_max = aminmax(x)

    torch.testing.assert_close(tri_min, ref_min, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_max, ref_max, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aminmax_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_min, ref_max = torch.aminmax(x, dim=dim, keepdim=keepdim)
    tri_min, tri_max = aminmax(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri_min, ref_min, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_max, ref_max, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aminmax_out(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    out_shape = list(shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)

    out_min = torch.empty(out_shape, device="cpu", dtype=torch.float32)
    out_max = torch.empty(out_shape, device="cpu", dtype=torch.float32)

    ref_min, ref_max = torch.aminmax(x, dim=dim, keepdim=keepdim)
    aminmax(x, dim=dim, keepdim=keepdim, out=(out_min, out_max))

    torch.testing.assert_close(out_min, ref_min, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_max, ref_max, rtol=1e-4, atol=1e-4)
