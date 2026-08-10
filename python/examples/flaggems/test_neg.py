import pytest
import torch

from .neg import neg, neg_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_neg(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = torch.neg(x)
    tri = neg(x)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_neg_inplace():
    x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    torch.neg(x_ref, out=x_ref)
    neg_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
