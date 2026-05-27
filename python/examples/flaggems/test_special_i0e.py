import pytest
import torch

from .special_i0e import special_i0e, special_i0e_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_i0e(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    ref_out = torch.special.i0e(x)
    tri_out = special_i0e(x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_i0e_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    out = torch.empty_like(x)
    ref_out = torch.special.i0e(x)
    special_i0e_out(x, out)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_i0e_fp16(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float16)
    ref_out = torch.special.i0e(x)
    tri_out = special_i0e(x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
