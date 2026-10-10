import pytest
import torch

from .subtract_ import subtract_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_subtract_(shape, alpha):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()

    x_ref.subtract_(y, alpha=alpha)
    subtract_(x, y, alpha=alpha)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
