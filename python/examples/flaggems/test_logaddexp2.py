import pytest
import torch

from .logaddexp2 import logaddexp2, logaddexp2_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_logaddexp2(size):
    torch.manual_seed(0)
    # Wide range including large negatives to exercise the numerically
    # stable form m + log2(1 + 2^-|x-y|).
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 50.0
    y = torch.randn(size, device="cpu", dtype=torch.float32) * 50.0

    ref = torch.logaddexp2(x, y)
    tri = logaddexp2(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_logaddexp2_scalar():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32) * 50.0

    ref = torch.logaddexp2(x, torch.tensor(0.0))
    tri = logaddexp2(x, 0.0)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_logaddexp2_fp64_fallback():
    # fp64 keeps full precision via torch; this test pins that fallback.
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float64) * 50.0
    y = torch.randn(1024, device="cpu", dtype=torch.float64) * 50.0

    ref = torch.logaddexp2(x, y)
    tri = logaddexp2(x, y)

    torch.testing.assert_close(tri, ref)


def test_logaddexp2_out():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32) * 50.0
    y = torch.randn(1024, device="cpu", dtype=torch.float32) * 50.0

    ref = torch.logaddexp2(x, y)
    out = torch.empty_like(x)
    ret = logaddexp2_out(x, y, out)

    assert ret is out
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_logaddexp2_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randn(shape_a) * 50.0
    b = torch.randn(shape_b) * 50.0

    ref_out = torch.logaddexp2(a, b)
    tri_out = logaddexp2(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@BF16_XFAIL
def test_logaddexp2_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 10.0).to(torch.bfloat16)
    y = (torch.randn(128, device="cpu", dtype=torch.float32) * 10.0).to(torch.bfloat16)

    ref = torch.logaddexp2(x, y)
    tri = logaddexp2(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


def test_logaddexp2_infinities():
    inf = float("inf")
    x = torch.tensor([inf, inf, -inf, -inf, 1.0, float("nan")])
    y = torch.tensor([inf, -inf, -inf, inf, inf, 1.0])

    ref = torch.logaddexp2(x, y)
    tri = logaddexp2(x, y)

    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_logaddexp2_fp16():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float16)
    y = torch.randn(128, dtype=torch.float16)
    torch.testing.assert_close(
        logaddexp2(x, y), torch.logaddexp2(x, y), rtol=1e-3, atol=1e-3
    )


def test_logaddexp2_mixed_dtype_promotes():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0], dtype=torch.float64)
    out = logaddexp2(x, y)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, torch.logaddexp2(x, y))
