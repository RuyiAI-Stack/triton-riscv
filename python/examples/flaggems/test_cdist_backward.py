import pytest
import torch

from .cdist_backward import _cdist_backward


@pytest.mark.parametrize("shape", [(2, 3), (4, 128), (16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cdist_backward_matches_autograd(shape, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=dtype, device="cpu", requires_grad=True)
    x2 = torch.randn((shape[0] * 2, shape[1]), dtype=dtype, device="cpu")

    dist = torch.cdist(x1, x2, p=2.0)
    grad = torch.randn_like(dist)

    x1_ref = x1.detach().clone().requires_grad_(True)
    dist_ref = torch.cdist(x1_ref, x2, p=2.0)
    dist_ref.backward(grad)
    ref_out = x1_ref.grad

    tri_out = _cdist_backward(grad, x1.detach(), x2, 2.0, dist.detach())

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("shape", [(3, 5), (4, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cdist_backward_batch_and_grad_shape(batch, shape, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(
        (batch, shape[0], shape[1]), dtype=dtype, device="cpu", requires_grad=True
    )
    x2 = torch.randn((batch, shape[0] + 1, shape[1]), dtype=dtype, device="cpu")

    dist = torch.cdist(x1, x2, p=2.0)
    grad = torch.randn_like(dist)

    x1_ref = x1.detach().clone().requires_grad_(True)
    dist_ref = torch.cdist(x1_ref, x2, p=2.0)
    dist_ref.backward(grad)
    ref_out = x1_ref.grad.squeeze(0) if batch == 1 else x1_ref.grad

    tri_out = _cdist_backward(grad, x1.detach(), x2, 2.0, dist)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batched", [False, True])
def test_cdist_backward_p0_is_zero(batched):
    torch.manual_seed(0)
    if batched:
        x1 = torch.randn((2, 3, 4), dtype=torch.float32, device="cpu")
        x2 = torch.randn((2, 5, 4), dtype=torch.float32, device="cpu")
        grad = torch.randn((2, 3, 5), dtype=torch.float32, device="cpu")
    else:
        x1 = torch.randn((3, 4), dtype=torch.float32, device="cpu")
        x2 = torch.randn((5, 4), dtype=torch.float32, device="cpu")
        grad = torch.randn((3, 5), dtype=torch.float32, device="cpu")

    dist = torch.cdist(x1, x2, p=0.0)
    out = _cdist_backward(grad, x1, x2, 0.0, dist)

    torch.testing.assert_close(out, torch.zeros_like(x1))
