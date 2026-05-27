import pytest
import torch

from .sum import sum, sum_dim, sum_dim_out, sum_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_sum(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.sum(x)
    tri = sum(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_sum_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.sum(x)
    out = torch.empty((), dtype=torch.float32, device="cpu")
    sum_out(x, out=out)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 32), (8, 128), (4, 256)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.sum(x, dim=dim, keepdim=keepdim)
    tri = sum_dim(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 32), (8, 128)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum_dim_out(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.sum(x, dim=dim, keepdim=keepdim)
    if keepdim:
        out_shape = list(shape)
        out_shape[dim] = 1
    else:
        out_shape = list(shape)
        out_shape.pop(dim)
    out = torch.empty(out_shape, dtype=torch.float32, device="cpu")
    sum_dim_out(x, dim=dim, keepdim=keepdim, out=out)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 3, 16, 32)])
@pytest.mark.parametrize("dim", [(0, 2), (1, 3)])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum_multi_dim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.sum(x, dim=dim, keepdim=keepdim)
    tri = sum_dim(x, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_sum_empty():
    x = torch.randn(0, 5, dtype=torch.float32, device="cpu")
    ref = torch.sum(x)
    tri = sum_dim(x)
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sum_int(dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 100, (512,), dtype=dtype, device="cpu")

    ref = torch.sum(x, dtype=dtype)
    tri = sum(x)

    torch.testing.assert_close(tri, ref)


def test_sum_dim_empty():
    x = torch.randn(0, 5, dtype=torch.float32, device="cpu")
    ref = torch.sum(x, dim=1)
    tri = sum_dim(x, dim=1)
    torch.testing.assert_close(tri, ref)


def test_sum_dim_none():
    x = torch.randn(2, 3, dtype=torch.float32, device="cpu")
    ref = torch.sum(x)
    tri = sum_dim(x)
    torch.testing.assert_close(tri, ref)


def test_sum_dim_empty_list():
    x = torch.randn(2, 3, dtype=torch.float32, device="cpu")
    ref = torch.sum(x)
    tri = sum_dim(x, dim=[])
    torch.testing.assert_close(tri, ref)


def test_sum_bool():
    x = torch.tensor([True, False, True], device="cpu")
    ref = torch.sum(x)
    tri = sum(x)
    torch.testing.assert_close(tri, ref)
