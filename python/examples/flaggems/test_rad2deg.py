import pytest
import torch

from .rad2deg import rad2deg


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_rad2deg(size):
    torch.manual_seed(0)
    x = (
        torch.rand(size, device="cpu", dtype=torch.float32) * 6.283185307179586
        - 3.141592653589793
    )

    ref_out = torch.rad2deg(x)
    tri_out = rad2deg(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_rad2deg_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128) * 6.283185307179586 - 3.141592653589793).to(dtype)

    ref_out = torch.rad2deg(x)
    tri_out = rad2deg(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_rad2deg_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 6.283185307179586 - 3.141592653589793).to(torch.bfloat16)

    ref_out = torch.rad2deg(x)
    tri_out = rad2deg(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
