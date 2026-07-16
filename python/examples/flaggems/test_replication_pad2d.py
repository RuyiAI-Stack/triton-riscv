import pytest
import torch

from .replication_pad2d import replication_pad2d, replication_pad2d_out


@pytest.mark.parametrize("shape", [(1, 1, 16, 32), (1, 1, 31, 33), (1, 1, 32, 32)])
@pytest.mark.parametrize("padding", [(1, 1, 1, 1), (2, 0, 1, 2), (0, 0, 0, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_replication_pad2d_forward(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cpu")

    out_triton = replication_pad2d(inp, padding)
    out_torch = torch.nn.functional.pad(inp, padding, mode="replicate")

    torch.testing.assert_close(out_triton, out_torch)


@pytest.mark.parametrize("shape", [(1, 1, 16, 16)])
@pytest.mark.parametrize("padding", [(1, 1, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_replication_pad2d_out(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cpu")

    pad_left, pad_right, pad_top, pad_bottom = padding
    if len(shape) == 4:
        N, C, H, W = shape
        out_shape = (N, C, H + pad_top + pad_bottom, W + pad_left + pad_right)
    else:
        C, H, W = shape
        out_shape = (C, H + pad_top + pad_bottom, W + pad_left + pad_right)

    out_triton = torch.empty(out_shape, dtype=dtype, device="cpu")
    replication_pad2d_out(inp, padding, out=out_triton)
    out_torch = torch.nn.functional.pad(inp, padding, mode="replicate")

    torch.testing.assert_close(out_triton, out_torch)
