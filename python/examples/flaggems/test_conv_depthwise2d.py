import pytest
import torch

from .conv_depthwise2d import _conv_depthwise2d


@pytest.mark.parametrize(
    "N, C, H, W, out_channels, kernel_size, stride, padding, dilation",
    [
        (2, 4, 16, 16, 4, 3, 1, 1, 1),
        (1, 3, 32, 32, 3, 5, 2, 2, 1),
        (2, 4, 16, 16, 8, 3, 1, 1, 2),
        (1, 8, 16, 16, 16, 3, 1, 1, 1),
        (1, 4, 16, 16, 4, 3, 1, 0, 1),
        (2, 4, 16, 16, 4, 3, 1, "same", 1),
        (2, 4, 16, 16, 4, 3, 1, "valid", 1),
        # Test required sizes: 512, 1023, 1024
        (1, 1, 512, 512, 1, 3, 1, 1, 1),
        (1, 1, 1023, 1023, 1, 3, 1, 1, 1),
        (1, 1, 1024, 1024, 1, 3, 1, 1, 1),
    ],
)
@pytest.mark.parametrize("use_bias", [True, False])
def test_conv_depthwise2d(
    N,
    C,
    H,
    W,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    use_bias,
):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(
        (N, C, H, W), dtype=torch.float32, device=device, requires_grad=True
    )
    weight = torch.randn(
        (out_channels, 1, kernel_size, kernel_size),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    if use_bias:
        bias = torch.randn(
            (out_channels,),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
    else:
        bias = None

    x_ref = x.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)
    if use_bias:
        bias_ref = bias.clone().detach().requires_grad_(True)
    else:
        bias_ref = None

    tri_out = _conv_depthwise2d(
        x,
        weight,
        [kernel_size, kernel_size],
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    if padding == "same" or padding == "valid":
        ref_out = torch.nn.functional.conv2d(
            x_ref,
            weight_ref,
            bias=bias_ref,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=C,
        )
    else:
        ref_out = torch.nn.functional.conv2d(
            x_ref,
            weight_ref,
            bias=bias_ref,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=C,
        )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)

    grad_out = torch.randn_like(tri_out)
    tri_out.backward(grad_out)
    ref_out.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        weight.grad, weight_ref.grad, rtol=1e-3, atol=1e-3
    )

    if use_bias:
        torch.testing.assert_close(
            bias.grad, bias_ref.grad, rtol=1e-3, atol=1e-3
        )
