import pytest
import torch

from .l1_loss import l1_loss

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_l1_loss(size, reduction):
    torch.manual_seed(0)
    x = torch.randn(size) * 3.0
    y = torch.randn(size) * 3.0

    ref = torch.nn.functional.l1_loss(x, y, reduction=reduction)
    tri = l1_loss(x, y, reduction=reduction)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_l1_loss_fp16(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128) * 3.0).to(dtype)
    y = (torch.randn(128) * 3.0).to(dtype)

    ref = torch.nn.functional.l1_loss(x, y)
    tri = l1_loss(x, y)

    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


def test_l1_loss_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float64)
    y = torch.randn(256, dtype=torch.float64)

    ref = torch.nn.functional.l1_loss(x, y)
    tri = l1_loss(x, y)

    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_l1_loss_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128) * 3.0).to(torch.bfloat16)
    y = (torch.randn(128) * 3.0).to(torch.bfloat16)

    ref = torch.nn.functional.l1_loss(x, y)
    tri = l1_loss(x, y)

    torch.testing.assert_close(tri.float(), ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape_x, shape_y", [((2, 3), (1, 3)), ((2, 3), (2, 1))])
def test_l1_loss_broadcast(shape_x, shape_y):
    torch.manual_seed(0)
    x = torch.randn(shape_x)
    y = torch.randn(shape_y)
    torch.testing.assert_close(l1_loss(x, y), torch.nn.functional.l1_loss(x, y))


def test_l1_loss_int_raises_like_torch():
    x = torch.tensor([1, 2])
    y = torch.tensor([3, 4])
    with pytest.raises(NotImplementedError):
        l1_loss(x, y)
