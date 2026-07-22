import pytest
import torch

from .negative import negative


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_negative(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.negative(x)
    out = negative(x)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
