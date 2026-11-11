import pytest
import torch

from .sigmoid_ import sigmoid_


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_sigmoid_inplace(size):
    torch.manual_seed(0)
    x = torch.rand(size, device="cpu", dtype=torch.float32) * 16.0 - 8.0

    ref = x.clone()
    torch.sigmoid_(ref)

    tri = x.clone()
    out = sigmoid_(tri)

    assert out is tri, "sigmoid_ should return the same tensor"
    torch.testing.assert_close(tri, ref.t(), rtol=1e-4, atol=1e-4)


def test_sigmoid_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.rand(32, 64, device="cpu", dtype=torch.float32) * 16.0 - 8.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.sigmoid_(ref)

    # .t() keeps the view genuinely non-contiguous (a .clone() of a
    # strided view re-packs contiguous, which would only exercise
    # the fast path).
    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = sigmoid_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_sigmoid_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.rand(128, device="cpu", dtype=torch.float32) * 16.0 - 8.0).to(dtype)

    ref = x.clone()
    torch.sigmoid_(ref)

    tri = x.clone()
    out = sigmoid_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t(), rtol=1e-3, atol=1e-3)


@BF16_XFAIL
def test_sigmoid__bf16():
    torch.manual_seed(0)
    x = (torch.rand(128) * 16.0 - 8.0).to(torch.bfloat16)

    ref = torch.sigmoid(x)

    tri = x.clone()
    out = sigmoid_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t(), rtol=1e-2, atol=1e-2)
