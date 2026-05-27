import pytest
import torch

from .all import all, all_dim, all_dims


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_all(size):
    torch.manual_seed(0)
    x = torch.ones(size, device="cpu", dtype=torch.float32)

    ref_out = torch.all(x)
    tri_out = all(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_all_false(size):
    torch.manual_seed(0)
    x = torch.ones(size, device="cpu", dtype=torch.float32)
    x[size // 2] = 0.0

    ref_out = torch.all(x)
    tri_out = all(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_all_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.ones(shape, device="cpu", dtype=torch.float32)
    x[0, 0] = 0.0  # introduce a zero

    ref_out = torch.all(x, dim=dim, keepdim=keepdim)
    tri_out = all_dim(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(4, 8, 16)])
@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (-1, -2)])
@pytest.mark.parametrize("keepdim", [True, False])
def test_all_dims(shape, dims, keepdim):
    torch.manual_seed(0)
    x = torch.ones(shape, device="cpu", dtype=torch.float32)
    x[0, 0, 0] = 0.0  # introduce a zero

    # PyTorch's torch.all does not natively support tuple of dims in standard api
    # but torch.amax works similarly for boolean contexts or we can sequentially apply it
    ref_out = x != 0
    for d in sorted([dim % x.ndim for dim in dims], reverse=True):
        ref_out = torch.all(ref_out, dim=d, keepdim=keepdim)

    tri_out = all_dims(x, dim=dims, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out)
