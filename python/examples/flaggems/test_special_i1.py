import pytest
import torch

from .special_i1 import special_i1, special_i1_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_special_i1(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = torch.special.i1(x)
    tri_out = special_i1(x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_special_i1_2d(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref_out = torch.special.i1(x)
    tri_out = special_i1(x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_special_i1_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    out = torch.empty(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.special.i1(x)
    tri_out = special_i1_out(x, out=out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
