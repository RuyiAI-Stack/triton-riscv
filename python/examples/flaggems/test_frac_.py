import pytest
import torch

from .frac_ import frac_


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_frac_inplace(size):
    torch.manual_seed(0)
    # Fractional values with mixed signs: frac(x) = x - trunc(x) is negative
    # for negative inputs, so the domain must include them.
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 5.0

    ref = x.clone()
    torch.frac_(ref)

    tri = x.clone()
    out = frac_(tri)

    assert out is tri, "frac_ should return the same tensor"
    torch.testing.assert_close(tri, ref.t())


def test_frac_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32) * 5.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.frac_(ref)

    # .t() keeps the view genuinely non-contiguous (a .clone() of a
    # strided view re-packs contiguous, which would only exercise
    # the fast path).
    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = frac_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_frac_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 5.0).to(dtype)

    ref = x.clone()
    torch.frac_(ref)

    tri = x.clone()
    out = frac_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t())


@BF16_XFAIL
def test_frac__bf16():
    torch.manual_seed(0)
    x = (torch.randn(128) * 5.0).to(torch.bfloat16)

    ref = torch.frac(x)

    tri = x.clone()
    out = frac_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)
