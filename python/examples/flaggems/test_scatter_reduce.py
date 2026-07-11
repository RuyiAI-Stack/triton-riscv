import pytest
import torch

from .scatter_reduce import scatter_reduce


@pytest.mark.parametrize(
    "shape, dim, reduce",
    [
        ((4, 8), 0, "sum"),
        ((4, 8), 1, "sum"),
        ((4, 8), 0, "amax"),
        ((4, 8), 1, "amin"),
    ],
)
def test_scatter_reduce(shape, dim, reduce):
    torch.manual_seed(0)
    inp = torch.randn(*shape, dtype=torch.float32)
    src = torch.randn(*shape, dtype=torch.float32)
    index = torch.randint(0, shape[dim], (shape[dim],), dtype=torch.long)
    # For non-dim reduction, index matches src shape
    index_expanded = index
    if dim == 0:
        index_expanded = index.unsqueeze(1).expand_as(src)
    elif dim == 1:
        index_expanded = index.unsqueeze(0).expand_as(src)

    ref = torch.scatter_reduce(inp.clone(), dim, index_expanded, src, reduce=reduce)
    tri = scatter_reduce(inp.clone(), dim, index_expanded, src, reduce=reduce)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reduce", ["sum", "mean", "amax", "amin"])
def test_scatter_reduce_partial_index_and_larger_src(dim, reduce):
    torch.manual_seed(0)
    inp = torch.randn((3, 5), dtype=torch.float32)
    index = torch.empty((2, 4), dtype=torch.int64)
    index.random_(0, inp.shape[dim])
    src = torch.randn((3, 5), dtype=torch.float32)

    ref = torch.scatter_reduce(inp, dim, index, src, reduce=reduce)
    tri = scatter_reduce(inp, dim, index, src, reduce=reduce)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)
