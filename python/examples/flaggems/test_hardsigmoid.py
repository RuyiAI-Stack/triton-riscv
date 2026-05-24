import pytest
import torch
import torch.nn.functional as F

from .hardsigmoid import hardsigmoid


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_hardsigmoid(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = F.hardsigmoid(x)
    tri = hardsigmoid(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_hardsigmoid_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = F.hardsigmoid(x)
    out = torch.empty_like(x)
    from .hardsigmoid import hardsigmoid_out

    tri = hardsigmoid_out(x, out)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
