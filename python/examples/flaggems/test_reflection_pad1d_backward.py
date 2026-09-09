import torch
import torch.nn.functional as F

from .reflection_pad1d_backward import reflection_pad1d_backward


def test_reflection_pad1d_backward():
    x = torch.randn(1, 2, 5, dtype=torch.float32, device="cpu", requires_grad=True)
    out = F.pad(x, (1, 2), mode="reflect")
    grad = torch.randn_like(out)
    out.backward(grad)

    tri = reflection_pad1d_backward(grad, x.detach(), (1, 2))

    torch.testing.assert_close(tri, x.grad, rtol=1e-4, atol=1e-4)
