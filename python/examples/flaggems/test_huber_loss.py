import pytest
import torch

from .huber_loss import huber_loss

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_huber_loss(size, reduction):
    torch.manual_seed(0)
    x = torch.randn(size) * 5.0
    y = torch.randn(size) * 5.0

    ref = torch.nn.functional.huber_loss(x, y, reduction=reduction)
    tri = huber_loss(x, y, reduction=reduction)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("delta", [0.5, 2.0])
def test_huber_loss_delta(delta):
    torch.manual_seed(0)
    x = torch.randn(1024) * 5.0
    y = torch.randn(1024) * 5.0

    ref = torch.nn.functional.huber_loss(x, y, delta=delta)
    tri = huber_loss(x, y, delta=delta)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_huber_loss_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float64) * 5.0
    y = torch.randn(256, dtype=torch.float64) * 5.0

    ref = torch.nn.functional.huber_loss(x, y)
    tri = huber_loss(x, y)

    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_huber_loss_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128) * 5.0).to(torch.bfloat16)
    y = (torch.randn(128) * 5.0).to(torch.bfloat16)

    ref = torch.nn.functional.huber_loss(x, y)
    tri = huber_loss(x, y)

    torch.testing.assert_close(tri.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_huber_loss_fp16():
    torch.manual_seed(0)
    x = (torch.randn(512) * 5.0).to(torch.float16)
    y = (torch.randn(512) * 5.0).to(torch.float16)
    torch.testing.assert_close(
        huber_loss(x, y),
        torch.nn.functional.huber_loss(x.float(), y.float()).to(torch.float16),
        rtol=1e-2,
        atol=1e-2,
    )


def test_huber_loss_nonpositive_delta_raises():
    x = torch.randn(16)
    y = torch.randn(16)
    with pytest.raises(RuntimeError):
        huber_loss(x, y, delta=0.0)
    with pytest.raises(RuntimeError):
        huber_loss(x, y, delta=-1.0)


def test_huber_loss_broadcast():
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    y = torch.randn(1, 3)
    torch.testing.assert_close(huber_loss(x, y), torch.nn.functional.huber_loss(x, y))


def test_huber_loss_int_raises_like_torch():
    x = torch.tensor([1, 2])
    y = torch.tensor([3, 4])
    with pytest.raises(NotImplementedError):
        huber_loss(x, y)
