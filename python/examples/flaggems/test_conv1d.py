import pytest
import torch

from .conv1d import conv1d


@pytest.mark.parametrize(
    "N, C, L, out_channels, kernel_size, stride, padding, dilation, groups",
    [
        (2, 4, 16, 8, 3, 1, 1, 1, 1),
        (1, 3, 32, 6, 5, 2, 2, 1, 1),
        (2, 4, 16, 8, 3, 1, 1, 2, 1),
        (1, 8, 16, 8, 3, 1, 1, 1, 2),
        (1, 4, 16, 8, 3, 1, 0, 1, 1),
        (2, 4, 16, 8, 3, 1, "same", 1, 1),
        (2, 4, 16, 8, 3, 1, "valid", 1, 1),
        # Test required sizes: 512, 1023, 1024
        (1, 1, 512, 1, 3, 1, 1, 1, 1),
        (1, 1, 1023, 1, 3, 1, 1, 1, 1),
        (1, 1, 1024, 1, 3, 1, 1, 1, 1),
    ],
)
@pytest.mark.parametrize("use_bias", [True, False])
def test_conv1d(
    N,
    C,
    L,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    use_bias,
):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(
        (N, C, L), dtype=torch.float32, device=device, requires_grad=True
    )
    weight = torch.randn(
        (out_channels, C // groups, kernel_size),
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

    tri_out = conv1d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    ref_out = torch.nn.functional.conv1d(
        x_ref,
        weight_ref,
        bias=bias_ref,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
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
