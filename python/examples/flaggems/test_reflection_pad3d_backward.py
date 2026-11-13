import pytest
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


@pytest.mark.parametrize("padding", [(1, 1, 0, 0, 0, 0), (0, 1, 1, 2, 1, 0)])
def test_reflection_pad3d_backward_asymmetric_padding(padding):
    x = torch.randn(1, 2, 2, 3, 4)
    padded = F.pad(x, padding, mode="reflect")
    grad = torch.arange(padded.numel(), dtype=x.dtype).reshape_as(padded)
    expected = torch.ops.aten.reflection_pad3d_backward(grad, x, padding)
    actual = reflection_pad3d_backward(grad, x, padding)
    torch.testing.assert_close(actual, expected)
