import pytest
import torch

from .cumsum_ import cumsum_


@pytest.mark.parametrize("shape", [(), (0,), (1023,), (3, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_cumsum_inplace(shape, dtype):
    x = torch.ones(shape, dtype=dtype)
    if x.ndim == 2:
        x = x.t()
    ref_out = x.clone().cumsum_(-1)
    assert cumsum_(x, -1) is x
    torch.testing.assert_close(x, ref_out)


@pytest.mark.parametrize("size", [16384, 16385])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.uint8, torch.int32])
def test_cumsum_inplace_block_prefix_boundary(size, dtype):
    x = torch.arange(size).remainder(7).to(dtype)
    ref_out = x.clone().cumsum_(0)
    pointer = x.data_ptr()
    assert cumsum_(x, 0) is x
    assert x.data_ptr() == pointer
    torch.testing.assert_close(x, ref_out, rtol=0, atol=0)
