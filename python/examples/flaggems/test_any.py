import pytest
import torch

from .any import any, any_dim, any_dims


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_any(size):
    torch.manual_seed(0)
    x = torch.zeros(size, device="cpu", dtype=torch.float32)

    ref_out = torch.any(x)
    tri_out = any(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_any_true(size):
    torch.manual_seed(0)
    x = torch.zeros(size, device="cpu", dtype=torch.float32)
    x[size // 2] = 1.0

    ref_out = torch.any(x)
    tri_out = any(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_any_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.zeros(shape, device="cpu", dtype=torch.float32)
    x[0, 0] = 1.0  # introduce a one

    ref_out = torch.any(x, dim=dim, keepdim=keepdim)
    tri_out = any_dim(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(4, 8, 16)])
@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (-1, -2)])
@pytest.mark.parametrize("keepdim", [True, False])
def test_any_dims(shape, dims, keepdim):
    torch.manual_seed(0)
    x = torch.zeros(shape, device="cpu", dtype=torch.float32)
    x[0, 0, 0] = 1.0  # introduce a one

    ref_out = torch.ne(x, 0)
    for d in sorted([dim % x.ndim for dim in dims], reverse=True):
        ref_out = torch.any(ref_out, dim=d, keepdim=keepdim)

    tri_out = any_dims(x, dim=dims, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out)
