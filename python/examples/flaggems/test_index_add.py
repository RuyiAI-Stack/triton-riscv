import pytest
import torch

from .index_add import index_add, index_add_


def _make_src(inp, dim, index):
    src_shape = list(inp.shape)
    src_shape[dim] = index.numel()
    return torch.randn(src_shape, dtype=inp.dtype, device=inp.device)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((16, 32), 0),
        ((16, 32), 1),
    ],
)
def test_index_add_2d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    index = torch.randint(0, shape[dim], (shape[dim] // 2,))
    src = _make_src(x, dim, index)
    ref_out = torch.index_add(x, dim, index, src, alpha=1.0)
    tri_out = index_add(x, dim, index, src, alpha=1.0)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((16, 32), 0),
    ],
)
def test_index_add_inplace_2d(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    index = torch.randint(0, shape[dim], (shape[dim] // 2,))
    src = _make_src(x, dim, index)
    x_ref.index_add_(dim, index, src, alpha=2.0)
    index_add_(x, dim, index, src, alpha=2.0)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_index_add_size_variants(size):
    torch.manual_seed(0)
    x = torch.randn((size,), dtype=torch.float32, device="cpu")
    idx_len = min(size // 4, 256)
    idx = torch.randint(0, size, (idx_len,))
    src = torch.randn((idx_len,), dtype=torch.float32, device="cpu")
    ref_out = torch.index_add(x, 0, idx, src, alpha=1.0)
    tri_out = index_add(x, 0, idx, src, alpha=1.0)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-3)
