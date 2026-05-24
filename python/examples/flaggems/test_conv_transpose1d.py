import pytest
import torch
import torch.nn.functional as F

from .conv_transpose1d import conv_transpose1d


@pytest.mark.parametrize(
    "batch, in_c, out_c, L, k",
    [
        (1, 4, 4, 8, 3),
        (2, 8, 4, 16, 3),
        (1, 4, 8, 10, 5),
    ],
)
def test_conv_transpose1d(batch, in_c, out_c, L, k):
    torch.manual_seed(0)
    x = torch.randn(batch, in_c, L, device="cpu", dtype=torch.float32)
    w = torch.randn(in_c, out_c, k, device="cpu", dtype=torch.float32)

    ref = F.conv_transpose1d(x, w)
    tri = conv_transpose1d(x, w)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "batch, in_c, out_c, L, k, stride, pad",
    [
        (1, 4, 4, 8, 3, 2, 1),
        (2, 8, 4, 16, 3, 1, 0),
    ],
)
def test_conv_transpose1d_stride_pad(batch, in_c, out_c, L, k, stride, pad):
    torch.manual_seed(0)
    x = torch.randn(batch, in_c, L, device="cpu", dtype=torch.float32)
    w = torch.randn(in_c, out_c, k, device="cpu", dtype=torch.float32)

    ref = F.conv_transpose1d(x, w, stride=stride, padding=pad)
    tri = conv_transpose1d(x, w, stride=stride, padding=pad)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_conv_transpose1d_bias():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8, device="cpu", dtype=torch.float32)
    w = torch.randn(4, 4, 3, device="cpu", dtype=torch.float32)
    bias = torch.randn(4, device="cpu", dtype=torch.float32)

    ref = F.conv_transpose1d(x, w, bias=bias)
    tri = conv_transpose1d(x, w, bias=bias)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
