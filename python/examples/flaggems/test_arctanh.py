import pytest
import torch

from .arctanh import arctanh


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arctanh(size):
    torch.manual_seed(0)
    # Open interval (-1, 1): arctanh diverges at the endpoints, so the sampled
    # domain stays strictly inside [-0.99, 0.99].
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 1.98 - 0.99

    ref_out = torch.arctanh(x)
    tri_out = arctanh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_arctanh_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128) * 1.98 - 0.99).to(dtype)

    ref_out = torch.arctanh(x)
    tri_out = arctanh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_arctanh_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 1.98 - 0.99).to(torch.bfloat16)

    ref_out = torch.arctanh(x)
    tri_out = arctanh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
