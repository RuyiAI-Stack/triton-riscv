import torch
import torch.nn.functional as F

from .reflection_pad3d_backward import reflection_pad3d_backward


def test_reflection_pad3d_backward():
    x = torch.randn(
        1, 1, 3, 3, 3, dtype=torch.float32, device="cpu", requires_grad=True
    )
    padding = (1, 1, 1, 1, 1, 1)
    out = F.pad(x, padding, mode="reflect")
    grad = torch.randn_like(out)
    out.backward(grad)

    tri = reflection_pad3d_backward(grad, x.detach(), padding)

    torch.testing.assert_close(tri, x.grad, rtol=1e-4, atol=1e-4)
