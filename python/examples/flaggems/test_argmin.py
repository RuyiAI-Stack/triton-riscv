import pytest
import torch

from .argmin import argmin


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_argmin_flattened(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.argmin(x)
    tri_out = argmin(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_argmin_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.argmin(x, dim=dim, keepdim=keepdim)
    tri_out = argmin(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(4, 8, 16)])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_argmin_3d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.argmin(x, dim=dim)
    tri_out = argmin(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out)


# argmin_kernel_opt_k1 path: K == 1 and not integer dtype
@pytest.mark.parametrize("shape", [(8, 64), (4, 128)])
@pytest.mark.parametrize("dim", [1])
def test_argmin_k1_opt(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.argmin(x, dim=dim)
    tri_out = argmin(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out)


# integer dtype fallback (skips opt_k1 and split_K paths)
@pytest.mark.parametrize("shape", [(4, 32), (8, 64)])
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_argmin_int(shape, dim, dtype):
    torch.manual_seed(0)
    x = torch.randint(-100, 100, shape, dtype=dtype)

    ref_out = torch.argmin(x, dim=dim)
    tri_out = argmin(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out)


# split_K kernel path: N%64==0, K%32==0, M%8==0, float
@pytest.mark.parametrize("shape", [(16, 64, 64)])
@pytest.mark.parametrize("dim", [1, 2])
def test_argmin_split_k(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.argmin(x, dim=dim)
    tri_out = argmin(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out)
