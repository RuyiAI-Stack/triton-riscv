import pytest
import torch

from .i0_ import i0_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_i0_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    x_ref = x.clone()
    x_ref_out = torch.i0(x_ref)
    i0_(x)
    torch.testing.assert_close(x, x_ref_out, rtol=1e-4, atol=1e-4)
