import torch
import torch.nn.functional as F

from .upsample_linear1d_backward import upsample_linear1d_backward


def test_upsample_linear1d_backward():
    x = torch.randn(1, 2, 5, dtype=torch.float32, device="cpu", requires_grad=True)
    out = F.interpolate(x, size=(9,), mode="linear", align_corners=False)
    grad = torch.randn_like(out)
    out.backward(grad)

    tri = upsample_linear1d_backward(
        grad, output_size=(9,), input_size=x.shape, align_corners=False
    )

    torch.testing.assert_close(tri, x.grad, rtol=1e-4, atol=1e-4)
