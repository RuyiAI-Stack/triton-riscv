import pytest
import torch

from .margin_ranking_loss import margin_ranking_loss


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
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_margin_ranking_loss(shape, reduction, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=dtype, device="cpu")
    x2 = torch.randn(shape, dtype=dtype, device="cpu")
    target = torch.sign(torch.randn(shape, dtype=dtype, device="cpu"))

    ref_out = torch.nn.functional.margin_ranking_loss(
        x1, x2, target, margin=1.0, reduction=reduction
    )
    tri_out = margin_ranking_loss(x1, x2, target, margin=1.0, reduction=reduction)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_margin_ranking_loss_none(shape, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=dtype, device="cpu")
    x2 = torch.randn(shape, dtype=dtype, device="cpu")
    target = torch.sign(torch.randn(shape, dtype=dtype, device="cpu"))

    ref_out = torch.nn.functional.margin_ranking_loss(
        x1, x2, target, margin=0.5, reduction="none"
    )
    tri_out = margin_ranking_loss(x1, x2, target, margin=0.5, reduction="none")

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2)
