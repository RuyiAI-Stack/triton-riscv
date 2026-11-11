import pytest
import torch

from .log2 import log2


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_log2(size):
    torch.manual_seed(0)
    # Domain (0, 100): log2 needs strictly positive inputs.
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 99.5 + 0.5

    ref_out = torch.log2(x)
    tri_out = log2(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_log2_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128) * 99.5 + 0.5).to(dtype)

    ref_out = torch.log2(x)
    tri_out = log2(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_log2_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 99.5 + 0.5).to(torch.bfloat16)

    ref_out = torch.log2(x)
    tri_out = log2(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
