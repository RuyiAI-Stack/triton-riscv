import pytest
import torch

from .hardsigmoid_ import hardsigmoid_

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_hardsigmoid_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 8.0

    ref = x.clone()
    torch.nn.functional.hardsigmoid(ref, inplace=True)

    tri = x.clone()
    out = hardsigmoid_(tri)

    assert out is tri, "hardsigmoid_ should return the same tensor"
    torch.testing.assert_close(tri, ref)


def test_hardsigmoid_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32) * 8.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.nn.functional.hardsigmoid(ref, inplace=True)

    tri = x_sliced.clone()
    out = hardsigmoid_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_hardsigmoid_inplace_nan():
    x = torch.tensor([-2.0, float("nan"), 4.0])

    ref = x.clone()
    torch.nn.functional.hardsigmoid(ref, inplace=True)

    tri = x.clone()
    out = hardsigmoid_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_hardsigmoid_inplace_fp16(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 8.0).to(dtype)

    ref = x.clone()
    torch.nn.functional.hardsigmoid(ref, inplace=True)

    tri = x.clone()
    out = hardsigmoid_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_hardsigmoid_inplace_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.float64) * 8.0

    ref = x.clone()
    torch.nn.functional.hardsigmoid(ref, inplace=True)

    tri = x.clone()
    out = hardsigmoid_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_hardsigmoid_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 8.0).to(torch.bfloat16)

    ref = x.clone()
    torch.nn.functional.hardsigmoid(ref, inplace=True)

    tri = x.clone()
    out = hardsigmoid_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)
