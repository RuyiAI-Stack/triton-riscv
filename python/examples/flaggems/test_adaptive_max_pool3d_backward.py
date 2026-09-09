import torch
import torch.nn.functional as F

from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward


def test_adaptive_max_pool3d_backward():
    torch.manual_seed(0)
    x = torch.randn(
        1, 2, 4, 4, 4, dtype=torch.float32, device="cpu", requires_grad=True
    )
    out, indices = F.adaptive_max_pool3d(x, (2, 2, 2), return_indices=True)
    grad = torch.randn_like(out)
    out.backward(grad)

    tri = adaptive_max_pool3d_backward(grad, x.detach(), indices)

    torch.testing.assert_close(tri, x.grad, rtol=1e-4, atol=1e-4)


def test_adaptive_max_pool3d_backward_non_contiguous_inputs_and_output_layout():
    torch.manual_seed(0)
    base = torch.randn(
        1, 2, 4, 4, 4, dtype=torch.float32, device="cpu", requires_grad=True
    )
    x = base.transpose(2, 4)
    x.retain_grad()

    out, indices = F.adaptive_max_pool3d(x, (2, 2, 2), return_indices=True)
    grad = torch.randn_like(out).transpose(2, 4)
    out.backward(grad)

    tri = adaptive_max_pool3d_backward(grad, x.detach(), indices)

    assert tri.stride() == x.stride()
    torch.testing.assert_close(tri, x.grad, rtol=1e-4, atol=1e-4)
