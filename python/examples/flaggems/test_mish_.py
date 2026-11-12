import pytest
import torch

from .mish_ import mish_

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_mish_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 3.0

    ref = x.clone()
    torch.nn.functional.mish(ref, inplace=True)

    tri = x.clone()
    out = mish_(tri)

    assert out is tri, "mish_ should return the same tensor"
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_mish_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32) * 3.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.nn.functional.mish(ref, inplace=True)

    tri = x_sliced.clone()
    out = mish_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@BF16_XFAIL
def test_mish_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 3.0).to(torch.bfloat16)

    ref = x.clone()
    torch.nn.functional.mish(ref, inplace=True)

    tri = x.clone()
    out = mish_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


def test_mish_inplace_fp64_fallback():
    # fp64 delegates to aten; this test pins that fallback.
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.float64) * 3.0

    ref = x.clone()
    torch.nn.functional.mish(ref, inplace=True)

    tri = x.clone()
    out = mish_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)
