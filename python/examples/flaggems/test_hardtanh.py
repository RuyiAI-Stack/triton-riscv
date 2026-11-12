import pytest
import torch

from .hardtanh import hardtanh, hardtanh_out

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: bfloat16 tl.maximum/minimum miscompile (RuyiAI-Stack/triton-riscv#118)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_hardtanh(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 3.0

    ref_out = torch.nn.functional.hardtanh(x)
    tri_out = hardtanh(x)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("min_val, max_val", [(-0.5, 2.0), (-10.0, -5.0)])
def test_hardtanh_custom_bounds(min_val, max_val):
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32) * 6.0

    ref_out = torch.nn.functional.hardtanh(x, min_val=min_val, max_val=max_val)
    tri_out = hardtanh(x, min_val=min_val, max_val=max_val)

    torch.testing.assert_close(tri_out, ref_out)


def test_hardtanh_int():
    torch.manual_seed(0)
    x = torch.randint(-10, 11, (1024,), dtype=torch.int32)

    ref_out = torch.nn.functional.hardtanh(x)
    tri_out = hardtanh(x)

    torch.testing.assert_close(tri_out, ref_out)


def test_hardtanh_out():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32) * 3.0

    ref_out = torch.nn.functional.hardtanh(x)
    out = torch.empty_like(x)
    ret = hardtanh_out(x, out=out)

    assert ret is out
    torch.testing.assert_close(out, ref_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_hardtanh_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 3.0).to(dtype)

    ref_out = torch.nn.functional.hardtanh(x)
    tri_out = hardtanh(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_hardtanh_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 3.0).to(torch.bfloat16)

    ref_out = torch.nn.functional.hardtanh(x)
    tri_out = hardtanh(x)

    torch.testing.assert_close(tri_out, ref_out)


def test_hardtanh_out_rejects_positional_out():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    out_t = torch.empty(4)
    with pytest.raises(TypeError):
        hardtanh_out(x, -1.0, 1.0, out_t)


def test_hardtanh_nan():
    x = torch.tensor([float("nan"), float("inf"), float("-inf"), 0.5])
    torch.testing.assert_close(
        hardtanh(x), torch.nn.functional.hardtanh(x), equal_nan=True
    )


def test_hardtanh_out_small_buffer_raises():
    x = torch.randn(32)
    out = torch.empty(16)
    with pytest.raises(ValueError):
        hardtanh_out(x, out=out)
