import pytest
import torch

from .log import log


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_log(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device="cpu") + 0.1

    ref_out = torch.log(x)
    tri_out = log(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
