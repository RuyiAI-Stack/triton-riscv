import pytest
import torch

from .hypot import hypot, hypot_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hypot(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    y = torch.randn(shape, dtype=dtype, device="cpu")

    ref_out = torch.hypot(x, y)
    tri_out = hypot(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hypot_out(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    y = torch.randn(shape, dtype=dtype, device="cpu")

    ref_out = torch.hypot(x, y)
    out = torch.empty_like(ref_out)
    tri_out = hypot_out(x, y, out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
