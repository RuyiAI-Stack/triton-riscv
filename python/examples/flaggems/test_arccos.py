import pytest
import torch

from .arccos import arccos


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arccos(size):
    torch.manual_seed(0)
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 2.0 - 1.0

    ref_out = torch.arccos(x)
    tri_out = arccos(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_arccos_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128) * 2.0 - 1.0).to(dtype)

    ref_out = torch.arccos(x)
    tri_out = arccos(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_arccos_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 2.0 - 1.0).to(torch.bfloat16)

    ref_out = torch.arccos(x)
    tri_out = arccos(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
