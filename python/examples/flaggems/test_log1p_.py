import pytest
import torch

from .log1p_ import log1p_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_log1p_inplace(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1
    x_ref = x.clone()

    x_ref.log1p_()
    log1p_(x)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
