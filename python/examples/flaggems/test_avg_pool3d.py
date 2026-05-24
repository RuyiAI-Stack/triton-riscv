import pytest
import torch
import torch.nn.functional as F

from .avg_pool3d import avg_pool3d, avg_pool3d_backward


@pytest.mark.parametrize(
    "kernel_size, stride, padding, count_include_pad, divisor_override",
    [
        (2, None, 0, True, None),
        ((3, 2, 2), (2, 1, 1), 1, True, None),
        (3, 1, 1, False, None),
        (3, 1, 1, False, 4),
    ],
)
def test_avg_pool3d_forward(
    kernel_size, stride, padding, count_include_pad, divisor_override
):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8, 8, device="cpu", dtype=torch.float32)

    ref_out = F.avg_pool3d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    tri_out = avg_pool3d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "kernel_size, stride, padding, count_include_pad, divisor_override",
    [
        (2, None, 0, True, None),
        ((3, 2, 2), (2, 1, 1), 1, True, None),
        (3, 1, 1, False, None),
    ],
)
def test_avg_pool3d_backward(
    kernel_size, stride, padding, count_include_pad, divisor_override
):
    torch.manual_seed(0)
    x = torch.randn(
        2, 3, 8, 8, 8, device="cpu", dtype=torch.float32, requires_grad=True
    )

    ref_out = F.avg_pool3d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    grad_output = torch.randn_like(ref_out)

    ref_out.backward(grad_output, retain_graph=True)
    ref_grad = x.grad.clone()

    tri_grad = avg_pool3d_backward(
        grad_output,
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-4, atol=1e-4)
