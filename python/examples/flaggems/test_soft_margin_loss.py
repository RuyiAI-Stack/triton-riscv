import pytest
import torch

from .soft_margin_loss import soft_margin_loss, soft_margin_loss_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,), (1_100_000,)])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_soft_margin_loss(shape, reduction):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randint(0, 2, shape, dtype=torch.float32, device="cpu") * 2 - 1

    ref_out = torch.nn.functional.soft_margin_loss(x, y, reduction=reduction)
    tri_out = soft_margin_loss(x, y, reduction=reduction)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_soft_margin_loss_out(shape, reduction):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randint(0, 2, shape, dtype=torch.float32, device="cpu") * 2 - 1

    ref_out = torch.nn.functional.soft_margin_loss(x, y, reduction=reduction)
    if reduction == "none":
        out = torch.empty(shape, dtype=torch.float32, device="cpu")
    else:
        out = torch.empty((), dtype=torch.float32, device="cpu")
    tri_out = soft_margin_loss_out(x, y, reduction=reduction, out=out)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
