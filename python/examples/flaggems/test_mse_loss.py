import pytest
import torch

from .mse_loss import mse_loss


@pytest.mark.parametrize(
    "shape, reduction",
    [
        ((16, 256), "mean"),
        ((4, 128), "sum"),
        ((512,), "mean"),
        ((1023,), "sum"),
        ((1024,), "mean"),
    ],
)
def test_mse_loss(shape, reduction):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.mse_loss(x, y, reduction=reduction)
    tri_out = mse_loss(x, y, reduction=reduction)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_mse_loss_none(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.mse_loss(x, y, reduction="none")
    tri_out = mse_loss(x, y, reduction="none")

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
