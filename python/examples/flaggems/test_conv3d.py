import pytest
import torch

from .conv3d import conv3d


@pytest.mark.parametrize(
    "N, C, D, H, W, out_channels, kernel_size, stride, padding, dilation, groups",
    [
        (1, 2, 4, 4, 4, 4, 3, 1, 1, 1, 1),
        (1, 2, 8, 8, 8, 4, 3, 2, 1, 1, 1),
        (1, 4, 8, 8, 8, 4, 3, 1, 1, 1, 2),
        (1, 2, 4, 4, 4, 4, 3, 1, "same", 1, 1),
        (1, 2, 4, 4, 4, 4, 3, 1, "valid", 1, 1),
    ],
)
@pytest.mark.parametrize("use_bias", [True, False])
def test_conv3d(
    N,
    C,
    D,
    H,
    W,
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

    x = torch.randn((N, C, D, H, W), dtype=torch.float32, device=device)
    weight = torch.randn(
        (out_channels, C // groups, kernel_size, kernel_size, kernel_size),
        dtype=torch.float32,
        device=device,
    )

    if use_bias:
        bias = torch.randn((out_channels,), dtype=torch.float32, device=device)
    else:
        bias = None

    tri_out = conv3d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    ref_out = torch.nn.functional.conv3d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
