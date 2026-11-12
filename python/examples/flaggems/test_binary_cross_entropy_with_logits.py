import pytest
import torch

from .binary_cross_entropy_with_logits import binary_cross_entropy_with_logits

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_bce_with_logits(size, reduction):
    torch.manual_seed(0)
    x = torch.randn(size) * 3.0
    t = torch.rand(size)

    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        x, t, reduction=reduction
    )
    tri = binary_cross_entropy_with_logits(x, t, reduction=reduction)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_bce_with_logits_extremes():
    # Large logits exercise the stable form's branches.
    x = torch.tensor([-100.0, -20.0, 0.0, 20.0, 100.0])
    t = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

    ref = torch.nn.functional.binary_cross_entropy_with_logits(x, t, reduction="none")
    tri = binary_cross_entropy_with_logits(x, t, reduction="none")

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


@BF16_XFAIL
def test_bce_with_logits_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128) * 3.0).to(torch.bfloat16)
    t = (torch.rand(128) > 0.5).float().to(torch.bfloat16)

    ref = torch.nn.functional.binary_cross_entropy_with_logits(x.float(), t.float()).to(
        torch.bfloat16
    )
    tri = binary_cross_entropy_with_logits(x, t)

    torch.testing.assert_close(tri.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_bce_with_logits_fp16():
    torch.manual_seed(0)
    x = (torch.randn(512) * 3.0).to(torch.float16)
    t = torch.rand(512).to(torch.float16)
    torch.testing.assert_close(
        binary_cross_entropy_with_logits(x, t),
        torch.nn.functional.binary_cross_entropy_with_logits(x.float(), t.float()).to(
            torch.float16
        ),
        rtol=1e-2,
        atol=1e-2,
    )


def test_bce_with_logits_inf_corners():
    # torch corners: (inf,0)->inf, (inf,1)->nan, (-inf,0)->nan, (-inf,1)->nan.
    # The kernel's x*(1-y) + softplus(-x) form reproduces all four.
    x = torch.tensor([float("inf"), float("inf"), float("-inf"), float("-inf")])
    t = torch.tensor([0.0, 1.0, 0.0, 1.0])
    tri = binary_cross_entropy_with_logits(x, t, reduction="none")
    ref = torch.nn.functional.binary_cross_entropy_with_logits(x, t, reduction="none")
    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_bce_with_logits_bad_reduction_raises():
    x = torch.randn(16)
    t = torch.rand(16)
    with pytest.raises(ValueError):
        binary_cross_entropy_with_logits(x, t, reduction="tot")


def test_bce_with_logits_shape_mismatch_raises_like_torch():
    x = torch.rand(2, 3)
    t = torch.rand(1, 3)
    with pytest.raises(ValueError):
        binary_cross_entropy_with_logits(x, t)


def test_bce_with_logits_mixed_dtype_promotes():
    x = torch.randn(8) * 3.0
    t = torch.rand(8, dtype=torch.float64)
    out = binary_cross_entropy_with_logits(x, t)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(x.double(), t)
    torch.testing.assert_close(out, ref)
