import pytest
import torch
import torch.nn.functional as F

from .cudnn_convolution import cudnn_convolution


@pytest.mark.parametrize(
    "N, C_in, C_out, D, H, W, kernel_size",
    [
        (1, 3, 6, 8, None, None, (3,)),
        (2, 3, 6, None, 16, 16, (3, 3)),
        (1, 3, 6, None, 8, 8, (3, 3)),
    ],
)
def test_cudnn_convolution(N, C_in, C_out, D, H, W, kernel_size):
    torch.manual_seed(0)
    ndim = len(kernel_size)
    if ndim == 1:
        x = torch.randn(N, C_in, D, device="cpu", dtype=torch.float32)
        w = torch.randn(C_out, C_in, *kernel_size, device="cpu", dtype=torch.float32)
        stride = (1,)
        padding = (0,)
        dilation = (1,)
        groups = 1
        ref = F.conv1d(
            x,
            w,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    elif ndim == 2:
        x = torch.randn(N, C_in, H, W, device="cpu", dtype=torch.float32)
        w = torch.randn(C_out, C_in, *kernel_size, device="cpu", dtype=torch.float32)
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)
        groups = 1
        ref = F.conv2d(
            x,
            w,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    tri = cudnn_convolution(
        x,
        w,
        padding,
        stride,
        dilation,
        groups,
        benchmark=True,
        deterministic=False,
        allow_tf32=False,
    )

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
