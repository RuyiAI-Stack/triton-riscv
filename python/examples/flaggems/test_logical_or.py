import pytest
import torch

from .logical_or import logical_or, logical_or_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_logical_or(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 2, shape, dtype=dtype, device="cpu")

    ref = x.logical_or(y)
    tri = logical_or(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


def test_logical_or_bool():
    x = torch.tensor([True, False, True, False], dtype=torch.bool, device="cpu")
    y = torch.tensor([True, True, False, False], dtype=torch.bool, device="cpu")
    ref = x.logical_or(y)
    tri = logical_or(x, y)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_logical_or_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    x_clone = x.clone()

    ref = x_clone.logical_or_(y)
    data_ptr = x.data_ptr()
    tri = logical_or_(x, y)

    assert tri.data_ptr() == data_ptr
    assert tri.dtype == dtype
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    torch.testing.assert_close(x, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logical_or_int32(shape):
    torch.manual_seed(0)
    x = torch.randint(0, 5, shape, dtype=torch.int32, device="cpu")
    y = torch.randint(0, 5, shape, dtype=torch.int32, device="cpu")

    ref = x.logical_or(y)
    tri = logical_or(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
