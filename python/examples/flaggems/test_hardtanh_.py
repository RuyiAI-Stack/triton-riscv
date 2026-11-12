import pytest
import torch

from .hardtanh_ import hardtanh_

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: bfloat16 tl.maximum/minimum miscompile (RuyiAI-Stack/triton-riscv#118)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_hardtanh_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 3.0

    ref = x.clone()
    torch.nn.functional.hardtanh(ref, inplace=True)

    tri = x.clone()
    out = hardtanh_(tri)

    assert out is tri, "hardtanh_ should return the same tensor"
    torch.testing.assert_close(tri, ref)


def test_hardtanh_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32) * 3.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.nn.functional.hardtanh(ref, inplace=True)

    tri = x_sliced.clone()
    out = hardtanh_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_hardtanh_inplace_custom_bounds():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32) * 6.0

    ref = x.clone()
    torch.nn.functional.hardtanh(ref, min_val=-0.5, max_val=2.0, inplace=True)

    tri = x.clone()
    out = hardtanh_(tri, min_val=-0.5, max_val=2.0)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_hardtanh_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 3.0).to(torch.bfloat16)

    ref = x.clone()
    torch.nn.functional.hardtanh(ref, inplace=True)

    tri = x.clone()
    out = hardtanh_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_hardtanh_inplace_fp64():
    x = torch.randn(64, dtype=torch.float64) * 3
    ref = x.clone()
    torch.testing.assert_close(hardtanh_(x), torch.nn.functional.hardtanh_(ref))
    torch.testing.assert_close(x, ref)


def test_hardtanh_inplace_int():
    x = torch.randint(-5, 5, (64,), dtype=torch.int32)
    ref = x.clone()
    torch.testing.assert_close(hardtanh_(x), torch.nn.functional.hardtanh_(ref))
    torch.testing.assert_close(x, ref)
