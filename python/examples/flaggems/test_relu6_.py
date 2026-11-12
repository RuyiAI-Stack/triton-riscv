import pytest
import torch

from .relu6_ import relu6_

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: bfloat16 tl.maximum/minimum miscompile (RuyiAI-Stack/triton-riscv#118)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_relu6_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 8.0

    ref = x.clone()
    torch.nn.functional.relu6(ref, inplace=True)

    tri = x.clone()
    out = relu6_(tri)

    assert out is tri, "relu6_ should return the same tensor"
    torch.testing.assert_close(tri, ref)


def test_relu6_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32) * 8.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.nn.functional.relu6(ref, inplace=True)

    tri = x_sliced.clone()
    out = relu6_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_relu6_inplace_int():
    torch.manual_seed(0)
    x = torch.randint(-10, 16, (1024,), dtype=torch.int32)

    ref = x.clone()
    torch.nn.functional.relu6(ref, inplace=True)

    tri = x.clone()
    out = relu6_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_relu6_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 8.0).to(dtype)

    ref = x.clone()
    ref.clamp_(0, 6)

    tri = x.clone()
    out = relu6_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_relu6_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 8.0).to(torch.bfloat16)

    ref = x.clone()
    ref.clamp_(0, 6)

    tri = x.clone()
    out = relu6_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)
