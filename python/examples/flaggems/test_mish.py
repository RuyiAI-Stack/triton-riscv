import pytest
import torch

from .mish import mish

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_mish(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 3.0

    ref_out = torch.nn.functional.mish(x)
    tri_out = mish(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_mish_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float64) * 3.0

    ref_out = torch.nn.functional.mish(x)
    tri_out = mish(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_mish_fp16(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 3.0).to(dtype)

    ref_out = torch.nn.functional.mish(x)
    tri_out = mish(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_mish_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 3.0).to(torch.bfloat16)

    ref_out = torch.nn.functional.mish(x)
    tri_out = mish(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


def test_mish_specials():
    # torch: mish(+inf) -> inf, mish(-inf) -> NaN, mish(NaN) -> NaN; pin the
    # shared behavior so an accidental divergence cannot slip through.
    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 1.5])
    tri = mish(x)
    ref = torch.nn.functional.mish(x)
    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_mish_int_raises_like_torch():
    with pytest.raises(NotImplementedError):
        mish(torch.tensor([1, -2], dtype=torch.int64))
