import pytest
import torch

from .acos import acos


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_acos(size):
    torch.manual_seed(0)
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 2.0 - 1.0

    ref_out = torch.acos(x)
    tri_out = acos(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
