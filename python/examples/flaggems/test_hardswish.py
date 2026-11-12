import pytest
import torch

from .hardswish import hardswish, hardswish_out

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_hardswish(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 6.0

    ref_out = torch.nn.functional.hardswish(x)
    tri_out = hardswish(x)

    torch.testing.assert_close(tri_out, ref_out)


def test_hardswish_out():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32) * 6.0

    ref_out = torch.nn.functional.hardswish(x)
    out = torch.empty_like(x)
    ret = hardswish_out(x, out)

    assert ret is out
    torch.testing.assert_close(out, ref_out)


def test_hardswish_nan():
    x = torch.tensor([-1.0, float("nan"), 3.0])

    ref_out = torch.nn.functional.hardswish(x)
    tri_out = hardswish(x)

    torch.testing.assert_close(tri_out, ref_out, equal_nan=True)


def test_hardswish_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float64) * 6.0

    ref_out = torch.nn.functional.hardswish(x)
    tri_out = hardswish(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_hardswish_fp16(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 6.0).to(dtype)

    ref_out = torch.nn.functional.hardswish(x)
    tri_out = hardswish(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_hardswish_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 6.0).to(torch.bfloat16)

    ref_out = torch.nn.functional.hardswish(x)
    tri_out = hardswish(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


def test_hardswish_int_raises_like_torch():
    with pytest.raises(NotImplementedError):
        hardswish(torch.tensor([1, -2], dtype=torch.int64))


def test_hardswish_out_small_buffer_raises():
    x = torch.randn(32)
    out = torch.empty(16)
    with pytest.raises(ValueError):
        hardswish_out(x, out)
