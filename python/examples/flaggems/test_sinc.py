import pytest
import torch

from .sinc import sinc


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sinc(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 2.0
    x[::7] = 0.0  # exact zeros exercise the removable singularity at 0

    ref_out = torch.sinc(x)
    tri_out = sinc(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_sinc_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128) * 4.0 - 2.0).to(dtype)

    ref_out = torch.sinc(x)
    tri_out = sinc(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_sinc_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 4.0 - 2.0).to(torch.bfloat16)

    ref_out = torch.sinc(x)
    tri_out = sinc(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


def test_sinc_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float64)
    torch.testing.assert_close(sinc(x), torch.sinc(x))
