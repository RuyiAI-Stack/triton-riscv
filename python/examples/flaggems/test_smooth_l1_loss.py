import pytest
import torch

from .smooth_l1_loss import (
    smooth_l1_loss,
    smooth_l1_loss_backward,
    smooth_l1_loss_out,
)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("beta", [0.0, 1.0])
@pytest.mark.parametrize("reduction_str", ["none", "mean", "sum"])
def test_smooth_l1_loss(size, beta, reduction_str):
    torch.manual_seed(0)
    input = torch.randn(size, dtype=torch.float32, device="cpu")
    target = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.smooth_l1_loss(
        input, target, reduction=reduction_str, beta=beta
    )
    tri_out = smooth_l1_loss(input, target, reduction=reduction_str, beta=beta)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_smooth_l1_loss_backward(size, beta):
    torch.manual_seed(0)
    input = torch.randn(
        size, dtype=torch.float32, device="cpu", requires_grad=True
    )
    target = torch.randn(size, dtype=torch.float32, device="cpu")

    loss = torch.nn.functional.smooth_l1_loss(
        input, target, reduction="mean", beta=beta
    )
    grad_output = torch.ones_like(loss)
    tri_grad = smooth_l1_loss_backward(
        grad_output, input.detach(), target, reduction="mean", beta=beta
    )

    loss.backward()
    ref_grad = input.grad

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1024])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_smooth_l1_loss_out(size, beta):
    torch.manual_seed(0)
    input = torch.randn(size, dtype=torch.float32, device="cpu")
    target = torch.randn(size, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.smooth_l1_loss(
        input, target, reduction="none", beta=beta
    )
    out = torch.empty(size, dtype=torch.float32, device="cpu")
    tri_out = smooth_l1_loss_out(
        input, target, reduction="none", beta=beta, out=out
    )
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
