import pytest
import torch

from .selu import selu


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_selu(size, dtype):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=dtype)
    ref_out = torch.nn.functional.selu(x)
    tri_out = selu(x)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
