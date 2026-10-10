import pytest
import torch

from .clamp_max import clamp_max, clamp_max_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_clamp_max(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu") * 4.0

    ref = torch.clamp_max(x, 1.25)
    out = clamp_max(x, 1.25)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_clamp_max_(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu") * 4.0
    x_ref = x.clone()

    x_ref.clamp_max_(1.25)
    clamp_max_(x, 1.25)

    torch.testing.assert_close(x, x_ref, rtol=1e-5, atol=1e-5)
