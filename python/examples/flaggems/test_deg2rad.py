import pytest
import torch

from .deg2rad import deg2rad


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_deg2rad(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu") * 180.0

    ref = torch.deg2rad(x)
    out = deg2rad(x)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
