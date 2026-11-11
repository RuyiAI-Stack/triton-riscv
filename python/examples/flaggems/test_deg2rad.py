import pytest
import torch

from .deg2rad import deg2rad


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_deg2rad(size):
    torch.manual_seed(0)
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 720.0 - 360.0

    ref_out = torch.deg2rad(x)
    tri_out = deg2rad(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_deg2rad_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128) * 720.0 - 360.0).to(dtype)

    ref_out = torch.deg2rad(x)
    tri_out = deg2rad(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_deg2rad_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 720.0 - 360.0).to(torch.bfloat16)

    ref_out = torch.deg2rad(x)
    tri_out = deg2rad(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
