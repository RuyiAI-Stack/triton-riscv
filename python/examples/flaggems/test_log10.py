import pytest
import torch

from .log10 import log10, log10_, log10_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_log10(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1

    ref_out = torch.log10(x)
    tri_out = log10(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,)])
def test_log10_inplace(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1
    x_ref = x.clone()

    x_ref.log10_()
    log10_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_log10_out(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1

    ref_out = torch.log10(x)

    out = torch.empty_like(x)
    tri_out = log10_out(x, out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
