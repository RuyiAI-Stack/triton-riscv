import pytest
import torch

from .sign import sign


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sign_float(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x[::7] = 0.0  # exact zeros must be part of the domain

    ref_out = torch.sign(x)
    tri_out = sign(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sign_int32(size):
    torch.manual_seed(0)
    x = torch.randint(-5, 6, (size,), dtype=torch.int32, device="cpu")

    ref_out = torch.sign(x)
    tri_out = sign(x)

    torch.testing.assert_close(tri_out, ref_out)


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@BF16_XFAIL
def test_sign_bf16():
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.bfloat16)

    ref_out = torch.sign(x)
    tri_out = sign(x)

    torch.testing.assert_close(tri_out, ref_out)


def test_sign_int64():
    torch.manual_seed(0)
    x = torch.randint(-5, 6, (128,), dtype=torch.int64, device="cpu")

    ref_out = torch.sign(x)
    tri_out = sign(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_sign_fp16_fp64(dtype):
    torch.manual_seed(0)
    x = torch.randn(128, dtype=dtype)
    torch.testing.assert_close(sign(x), torch.sign(x))
