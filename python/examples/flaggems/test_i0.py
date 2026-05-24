import pytest
import torch

from .i0 import i0, i0_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_i0(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref_out = torch.i0(x)
    tri_out = i0(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_i0_out(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    ref_out = torch.i0(x)
    out = torch.empty_like(ref_out)
    tri_out = i0_out(x, out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
