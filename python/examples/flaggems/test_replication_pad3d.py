import pytest
import torch

from .replication_pad3d import replication_pad3d, replication_pad3d_out


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8, 8), (1, 3, 10, 10, 10), (3, 8, 8, 8)]
)
@pytest.mark.parametrize(
    "padding", [(1, 1, 1, 1, 1, 1), (2, 0, 1, 2, 0, 1), (0, 0, 0, 0, 0, 0)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_replication_pad3d_forward(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cpu")

    out_triton = replication_pad3d(inp, padding)
    out_torch = torch.nn.functional.pad(inp, padding, mode="replicate")

    torch.testing.assert_close(out_triton, out_torch)


@pytest.mark.parametrize("shape", [(1, 1, 8, 8, 8)])
@pytest.mark.parametrize("padding", [(1, 1, 1, 1, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_replication_pad3d_out(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cpu")

    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = padding
    if len(shape) == 5:
        N, C, D, H, W = shape
        out_shape = (
            N,
            C,
            D + pad_front + pad_back,
            H + pad_top + pad_bottom,
            W + pad_left + pad_right,
        )
    else:
        C, D, H, W = shape
        out_shape = (
            C,
            D + pad_front + pad_back,
            H + pad_top + pad_bottom,
            W + pad_left + pad_right,
        )

    out_triton = torch.empty(out_shape, dtype=dtype, device="cpu")
    replication_pad3d_out(inp, padding, out=out_triton)
    out_torch = torch.nn.functional.pad(inp, padding, mode="replicate")

    torch.testing.assert_close(out_triton, out_torch)
