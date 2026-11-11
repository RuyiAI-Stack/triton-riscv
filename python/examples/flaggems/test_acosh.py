import pytest
import torch

from .acosh import acosh, acosh_out


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_acosh(size):
    torch.manual_seed(0)
    # acosh domain is [1, inf); sample values in [1, 10).
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 9.0 + 1.0

    ref_out = torch.acosh(x)
    tri_out = acosh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_acosh_out(size):
    torch.manual_seed(0)
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 9.0 + 1.0
    out = torch.empty_like(x)

    ref_out = torch.acosh(x)
    acosh_out(x, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_acosh_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128, device="cpu", dtype=torch.float32) * 9.0 + 1.0).to(dtype)

    ref_out = torch.acosh(x)
    tri_out = acosh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_acosh_bf16():
    torch.manual_seed(0)
    x = (torch.rand(128, device="cpu", dtype=torch.float32) * 9.0 + 1.0).to(
        torch.bfloat16
    )

    ref_out = torch.acosh(x)
    tri_out = acosh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


def test_acosh_out_dtype_mismatch():
    torch.manual_seed(0)
    x = torch.rand(128, device="cpu", dtype=torch.float32) * 9.0 + 1.0
    out = torch.empty(128, dtype=torch.float64)

    ref_out = torch.acosh(x)
    ret = acosh_out(x, out)

    assert ret is out, "acosh_out must return the caller's tensor"
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref_out.to(torch.float64), rtol=1e-4, atol=1e-4)
