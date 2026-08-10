import pytest
import torch

from .slice_scatter import slice_scatter


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_slice_scatter_1d(size):
    torch.manual_seed(0)
    inp = torch.randn(size, device="cpu", dtype=torch.float32)
    src = torch.randn(size // 2, device="cpu", dtype=torch.float32)
    ref_out = torch.ops.aten.slice_scatter(
        inp, src, dim=0, start=0, end=size // 2, step=1
    )
    tri_out = slice_scatter(inp, src, dim=0, start=0, end=size // 2, step=1)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_slice_scatter_2d(size):
    torch.manual_seed(0)
    inp = torch.randn(8, size, device="cpu", dtype=torch.float32)
    src = torch.randn(8, size // 2, device="cpu", dtype=torch.float32)
    ref_out = torch.ops.aten.slice_scatter(
        inp, src, dim=1, start=0, end=size // 2, step=1
    )
    tri_out = slice_scatter(inp, src, dim=1, start=0, end=size // 2, step=1)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_slice_scatter_with_step(size):
    torch.manual_seed(0)
    inp = torch.randn(size, device="cpu", dtype=torch.float32)
    half = (size + 1) // 2  # use ceiling division so src size matches the slice
    src = torch.randn(half, device="cpu", dtype=torch.float32)
    ref_out = torch.ops.aten.slice_scatter(
        inp, src, dim=0, start=0, end=half * 2, step=2
    )
    tri_out = slice_scatter(inp, src, dim=0, start=0, end=half * 2, step=2)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
