import pytest
import torch

from .logit_ import logit_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logit_inplace_no_eps(shape):
    torch.manual_seed(0)
    x = torch.sigmoid(torch.randn(shape, dtype=torch.float32, device="cpu"))
    x_ref = x.clone()

    x_ref.logit_()
    logit_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_logit_inplace_with_eps(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()

    x_ref.logit_(eps=1e-3)
    logit_(x, eps=1e-3)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
