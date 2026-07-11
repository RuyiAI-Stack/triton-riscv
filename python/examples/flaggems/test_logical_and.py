import pytest
import torch

from .logical_and import logical_and, logical_and_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_logical_and(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 2, shape, dtype=dtype, device="cpu")

    ref = torch.logical_and(x, y)
    tri = logical_and(x, y)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_logical_and_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    x_clone = x.clone()

    ref = x_clone.logical_and_(y)
    data_ptr = x.data_ptr()
    tri = logical_and_(x, y)

    assert tri.data_ptr() == data_ptr
    assert tri.dtype == dtype
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    torch.testing.assert_close(x, ref, rtol=0, atol=0)
