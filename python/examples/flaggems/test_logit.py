import pytest
import torch

from .logit import logit, logit_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logit_no_eps(shape):
    torch.manual_seed(0)
    x = torch.sigmoid(torch.randn(shape, dtype=torch.float32, device="cpu"))

    ref_out = torch.logit(x)
    tri_out = logit(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logit_with_eps(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.logit(x, eps=1e-3)
    tri_out = logit(x, eps=1e-3)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logit_out(shape):
    torch.manual_seed(0)
    x = torch.sigmoid(torch.randn(shape, dtype=torch.float32, device="cpu"))

    ref_out = torch.logit(x)

    out = torch.empty_like(x)
    tri_out = logit_out(x, out=out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logit_out_with_eps(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.logit(x, eps=1e-3)

    out = torch.empty_like(x)
    tri_out = logit_out(x, eps=1e-3, out=out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
