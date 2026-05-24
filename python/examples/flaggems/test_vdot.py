import pytest
import torch

from .vdot import vdot


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vdot_float32(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.vdot(x, y)
    tri_out = vdot(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vdot_int32(size):
    torch.manual_seed(0)
    x = torch.randint(-10, 10, (size,), device="cpu", dtype=torch.int32)
    y = torch.randint(-10, 10, (size,), device="cpu", dtype=torch.int32)

    ref_out = torch.vdot(x, y)
    tri_out = vdot(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
