import pytest
import torch

from .asin import asin


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_asin(size):
    torch.manual_seed(0)
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 2.0 - 1.0

    ref_out = torch.asin(x)
    tri_out = asin(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_asin_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128) * 2.0 - 1.0).to(dtype)

    ref_out = torch.asin(x)
    tri_out = asin(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_asin_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 2.0 - 1.0).to(torch.bfloat16)

    ref_out = torch.asin(x)
    tri_out = asin(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
