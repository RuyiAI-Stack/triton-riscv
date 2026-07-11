import pytest
import torch
import torch.nn.functional as F

from .max_pool2d_with_indices import (
    max_pool2d_backward,
    max_pool2d_with_indices,
)


@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, shape",
    [
        (2, None, 0, 1, (1, 1, 16, 16)),
        (3, 1, 0, 1, (1, 1, 16, 16)),
        (3, 2, 1, 1, (1, 1, 16, 16)),
        (2, 2, 0, 1, (1, 1, 512, 512)),
        (3, 1, 1, 1, (1, 1, 1023, 1023)),
        (3, 1, 1, 1, (1, 1, 1024, 1024)),
    ],
)
def test_max_pool2d_with_indices_forward(kernel_size, stride, padding, dilation, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cpu", dtype=torch.float32)

    ref_out, ref_idx = F.max_pool2d_with_indices(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=False,
    )
    tri_out, tri_idx = max_pool2d_with_indices(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=False,
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_idx, ref_idx, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, shape",
    [
        (2, 2, 0, 1, (1, 1, 8, 8)),
        (3, 1, 1, 1, (1, 1, 8, 8)),
    ],
)
def test_max_pool2d_backward(kernel_size, stride, padding, dilation, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cpu", dtype=torch.float32, requires_grad=True)

    ref_out, ref_idx = F.max_pool2d_with_indices(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output)

    tri_out, tri_idx = max_pool2d_with_indices(
        x.detach(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    tri_grad = max_pool2d_backward(
        grad_output,
        x.detach(),
        tri_idx,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=False,
    )

    torch.testing.assert_close(tri_grad, x.grad, rtol=1e-4, atol=1e-4)


def test_max_pool2d_preserves_int64_order_and_float64_gradient():
    base = 2**60
    integer_input = torch.tensor(
        [[[[base, base + 1], [base + 2, base + 3]]]], dtype=torch.int64
    )
    ref_out, ref_idx = F.max_pool2d_with_indices(integer_input, 2)
    tri_out, tri_idx = max_pool2d_with_indices(integer_input, 2)
    torch.testing.assert_close(tri_out, ref_out)
    torch.testing.assert_close(tri_idx, ref_idx)

    x = torch.randn((1, 1, 4, 4), dtype=torch.float64, requires_grad=True)
    ref, _ = F.max_pool2d_with_indices(x, 2)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    _, indices = max_pool2d_with_indices(x.detach(), 2)
    tri_grad = max_pool2d_backward(grad, x.detach(), indices, 2, None, 0, 1, False)
    assert tri_grad.dtype == torch.float64
    torch.testing.assert_close(tri_grad, x.grad)
