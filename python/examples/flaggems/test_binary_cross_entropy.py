import pytest
import torch

from .binary_cross_entropy import binary_cross_entropy

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_bce(size, reduction):
    torch.manual_seed(0)
    x = torch.sigmoid(torch.randn(size))
    t = torch.rand(size)

    ref = torch.nn.functional.binary_cross_entropy(x, t, reduction=reduction)
    tri = binary_cross_entropy(x, t, reduction=reduction)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_bce_extreme_probs():
    x = torch.tensor([1e-8, 1.0 - 1e-8, 0.5])
    t = torch.tensor([1.0, 1.0, 0.0])

    ref = torch.nn.functional.binary_cross_entropy(x, t, reduction="none")
    tri = binary_cross_entropy(x, t, reduction="none")

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_bce_fp64_fallback():
    torch.manual_seed(0)
    x = torch.sigmoid(torch.randn(256, dtype=torch.float64))
    t = torch.rand(256, dtype=torch.float64)

    ref = torch.nn.functional.binary_cross_entropy(x, t)
    tri = binary_cross_entropy(x, t)

    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_bce_bf16():
    torch.manual_seed(0)
    x = torch.sigmoid(torch.randn(128)).to(torch.bfloat16)
    t = (torch.rand(128) > 0.5).to(torch.bfloat16)

    ref = torch.nn.functional.binary_cross_entropy(x.float(), t.float()).to(
        torch.bfloat16
    )
    tri = binary_cross_entropy(x, t)

    torch.testing.assert_close(tri.float(), ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_bce_fp16(dtype):
    torch.manual_seed(0)
    x = torch.sigmoid(torch.randn(512)).to(dtype)
    t = torch.rand(512).to(dtype)
    torch.testing.assert_close(
        binary_cross_entropy(x, t),
        torch.nn.functional.binary_cross_entropy(x.float(), t.float()).to(dtype),
        rtol=1e-2,
        atol=1e-2,
    )


def test_bce_bad_reduction_raises():
    x = torch.sigmoid(torch.randn(16))
    t = torch.rand(16)
    with pytest.raises(ValueError):
        binary_cross_entropy(x, t, reduction="tot")


def test_bce_shape_mismatch_raises_like_torch():
    x = torch.rand(2, 3)
    t = torch.rand(1, 3)  # broadcastable but not same-shape
    with pytest.raises(ValueError):
        binary_cross_entropy(x, t)


def test_bce_mixed_dtype_raises_like_torch():
    x = torch.rand(4)
    t = torch.rand(4, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        binary_cross_entropy(x, t)
