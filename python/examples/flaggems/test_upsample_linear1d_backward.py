import pytest
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


@pytest.mark.parametrize("in_w,out_w", [(2, 10), (1, 10), (4, 1), (3, 8)])
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_linear1d_backward_support_bounds(in_w, out_w, align_corners):
    grad = torch.ones(1, 2, out_w)
    expected = torch.ops.aten.upsample_linear1d_backward(
        grad, [out_w], [1, 2, in_w], align_corners, None
    )
    actual = upsample_linear1d_backward(grad, [out_w], [1, 2, in_w], align_corners)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "in_w,out_w,scale", [(3, 5, 1.7), (3, 3, 1.1), (3, 5, 0.5), (3, 5, 10.0)]
)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("sequence_scale", [False, True])
def test_upsample_linear1d_backward_explicit_scale(
    in_w, out_w, scale, align_corners, sequence_scale
):
    grad = torch.arange(2 * out_w, dtype=torch.float32).reshape(1, 2, out_w)
    expected = torch.ops.aten.upsample_linear1d_backward(
        grad, [out_w], [1, 2, in_w], align_corners, scale
    )
    actual = upsample_linear1d_backward(
        grad,
        [out_w],
        [1, 2, in_w],
        align_corners,
        [scale] if sequence_scale else scale,
    )
    torch.testing.assert_close(actual, expected)


def test_upsample_linear1d_backward_inferred_output_size():
    x = torch.randn(1, 2, 3, requires_grad=True)
    output = F.interpolate(x, scale_factor=1.7, mode="linear", align_corners=False)
    grad = torch.arange(output.numel(), dtype=x.dtype).reshape_as(output)
    output.backward(grad)
    actual = upsample_linear1d_backward(grad, None, x.shape, False, [1.7])
    torch.testing.assert_close(actual, x.grad)
