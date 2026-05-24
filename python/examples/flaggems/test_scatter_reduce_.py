import pytest
import torch

from .scatter_reduce_ import scatter_reduce_


@pytest.mark.parametrize(
    "shape, dim, reduce",
    [
        ((4, 8), 1, "sum"),
        ((4, 8), 1, "amin"),
        ((4, 8), 1, "mean"),
    ],
)
def test_scatter_reduce_(shape, dim, reduce):
    torch.manual_seed(0)
    inp = torch.randn(*shape, dtype=torch.float32)
    src = torch.randn(*shape, dtype=torch.float32)
    index = torch.randint(0, shape[dim], src.shape, dtype=torch.long)

    inp_clone = inp.clone()
    ref = torch.scatter_reduce(inp.clone(), dim, index, src, reduce=reduce)
    tri = scatter_reduce_(inp_clone, dim, index, src, reduce=reduce)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_scatter_reduce_include_self_false():
    torch.manual_seed(0)
    inp = torch.randn(4, 8, dtype=torch.float32)
    src = torch.randn(4, 8, dtype=torch.float32)
    index = torch.randint(0, 8, src.shape, dtype=torch.long)

    inp_clone = inp.clone()
    ref = torch.scatter_reduce(
        inp.clone(), 1, index, src, reduce="sum", include_self=False
    )
    tri = scatter_reduce_(
        inp_clone, 1, index, src, reduce="sum", include_self=False
    )

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
