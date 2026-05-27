import pytest
import torch

from .logaddexp import logaddexp, logaddexp_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logaddexp(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.logaddexp(x, y)
    tri_out = logaddexp(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logaddexp_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.logaddexp(x, y)

    out = torch.empty_like(x)
    tri_out = logaddexp_out(x, y, out)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
