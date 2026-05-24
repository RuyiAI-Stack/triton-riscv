import pytest
import torch
import torch.nn.functional as F

from .conv_transpose2d import conv_transpose2d


@pytest.mark.parametrize(
    "batch, in_c, out_c, H, W, k",
    [
        (1, 4, 4, 8, 8, 3),
        (2, 4, 8, 6, 6, 4),
    ],
)
def test_conv_transpose2d(batch, in_c, out_c, H, W, k):
    torch.manual_seed(0)
    x = torch.randn(batch, in_c, H, W, device="cpu", dtype=torch.float32)
    w = torch.randn(in_c, out_c, k, k, device="cpu", dtype=torch.float32)

    ref = F.conv_transpose2d(x, w)
    tri = conv_transpose2d(x, w)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_conv_transpose2d_bias():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 8, 8, device="cpu", dtype=torch.float32)
    w = torch.randn(8, 4, 3, 3, device="cpu", dtype=torch.float32)
    bias = torch.randn(4, device="cpu", dtype=torch.float32)

    ref = F.conv_transpose2d(x, w, bias=bias)
    tri = conv_transpose2d(x, w, bias=bias)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
