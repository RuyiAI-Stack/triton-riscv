import pytest
import torch

from .fix_ import fix_


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_fix_inplace(size):
    torch.manual_seed(0)
    # Fractional values with mixed signs: fix truncates toward zero, which
    # only differs from floor on the negative side.
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 5.0

    ref = torch.fix(x)

    tri = x.clone()
    out = fix_(tri)

    assert out is tri, "fix_ should return the same tensor"
    torch.testing.assert_close(tri, ref.t())


def test_fix_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32) * 5.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = torch.fix(x_sliced)

    # .t() keeps the view genuinely non-contiguous (a .clone() of a
    # strided view re-packs contiguous, which would only exercise
    # the fast path).
    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = fix_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_fix_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 5.0).to(dtype)

    ref = torch.fix(x)

    tri = x.clone()
    out = fix_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t())


@BF16_XFAIL
def test_fix__bf16():
    torch.manual_seed(0)
    x = (torch.randn(128) * 5.0).to(torch.bfloat16)

    ref = torch.fix(x)

    tri = x.clone()
    out = fix_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)
