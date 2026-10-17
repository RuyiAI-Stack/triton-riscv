import pytest
import torch
import torch.nn.functional as F

from .soft_margin_loss_backward import soft_margin_loss_backward


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_soft_margin_loss_backward(shape, dtype, reduction):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    x_ref = x.clone().requires_grad_(True)
    target = torch.randint(0, 2, shape, device="cpu", dtype=dtype) * 2 - 1
    ref = F.soft_margin_loss(x_ref, target, reduction=reduction)

    if reduction == "none":
        grad_output = torch.ones_like(ref)
    else:
        grad_output = torch.ones((), dtype=dtype, device="cpu")

    ref.backward(grad_output)
    ref_out = x_ref.grad

    if reduction == "none":
        grad_output_tri = torch.ones_like(ref)
    else:
        grad_output_tri = grad_output.clone()

    tri_out = soft_margin_loss_backward(grad_output_tri, x, target, reduction=reduction)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_soft_margin_loss_backward_reduction_sum_scalar_grad(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    target = torch.randint(0, 2, shape, device="cpu", dtype=dtype) * 2 - 1
    reduction = "sum"
    grad_output = torch.tensor(2.5, dtype=dtype, device="cpu")

    x_ref = x.clone().requires_grad_(True)
    ref = F.soft_margin_loss(x_ref, target, reduction=reduction)
    ref.backward(grad_output)
    ref_out = x_ref.grad

    tri_out = soft_margin_loss_backward(grad_output, x, target, reduction=reduction)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
