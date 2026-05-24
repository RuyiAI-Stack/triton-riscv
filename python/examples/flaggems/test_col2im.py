import pytest
import torch
import torch.nn.functional as F

from .col2im import col2im


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("C", [2, 4])
@pytest.mark.parametrize(
    "out_size, kernel, stride",
    [
        ((4, 4), (2, 2), (1, 1)),
        ((4, 4), (2, 2), (2, 2)),
        ((8, 8), (3, 3), (1, 1)),
    ],
)
def test_col2im(batch, C, out_size, kernel, stride):
    torch.manual_seed(0)
    out_h, out_w = out_size
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    padding = 0
    dilation = 1

    x = torch.randn(batch, C, out_h, out_w, dtype=torch.float32, device="cpu")
    unfolded = F.unfold(
        x, kernel, dilation=dilation, padding=padding, stride=stride
    )

    ref = F.fold(
        unfolded,
        out_size,
        kernel,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    tri = col2im(unfolded, out_size, kernel, dilation, padding, stride)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
