import platform

import pytest
import torch

from .log1p import log1p


# The defect is an x86-host launcher ABI issue (the __truncsfbf2
# call-site register category); the riscv64 backend lowers bf16
# correctly, so the pin only applies where the defect reproduces.
if platform.machine() in ("x86_64", "AMD64"):
    BF16_XFAIL = pytest.mark.xfail(
        strict=True,
        reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
    )
else:

    def BF16_XFAIL(func):  # riscv64: the bare test must pass
        return func


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_log1p(size):
    torch.manual_seed(0)
    # Keep 1 + x well inside (0, inf): x in [-0.9, 9].
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 9.9 - 0.9

    ref_out = torch.log1p(x)
    tri_out = log1p(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_log1p_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128) * 9.9 - 0.9).to(dtype)

    ref_out = torch.log1p(x)
    tri_out = log1p(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_log1p_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 9.9 - 0.9).to(torch.bfloat16)

    ref_out = torch.log1p(x)
    tri_out = log1p(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
