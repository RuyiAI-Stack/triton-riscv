import pytest
import torch

from .scatter_reduce_ import scatter_reduce_


@pytest.mark.parametrize(
    "shape, dim, reduce",
    [
        ((4, 8), 1, "sum"),
        ((4, 8), 1, "prod"),
        ((3, 4, 5), 0, "prod"),
        ((4, 8), 1, "amax"),
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
    tri = scatter_reduce_(inp_clone, 1, index, src, reduce="sum", include_self=False)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_scatter_reduce_prod_include_self_false():
    torch.manual_seed(0)
    inp = torch.randn(4, 8, dtype=torch.float32)
    src = torch.randn(4, 8, dtype=torch.float32)
    index = torch.randint(0, 8, src.shape, dtype=torch.long)

    ref = torch.scatter_reduce(
        inp.clone(), 1, index, src, reduce="prod", include_self=False
    )
    tri = scatter_reduce_(inp.clone(), 1, index, src, reduce="prod", include_self=False)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_scatter_reduce_prod_partial_non_reduction_shape():
    torch.manual_seed(0)
    inp = torch.randn(4, 8, dtype=torch.float32)
    src = torch.randn(2, 8, dtype=torch.float32)
    index = torch.randint(0, 8, src.shape, dtype=torch.long)

    ref = torch.scatter_reduce(inp.clone(), 1, index, src, reduce="prod")
    tri = scatter_reduce_(inp.clone(), 1, index, src, reduce="prod")

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_prod_empty_reduction_dim(include_self):
    inp = torch.randn(3, 4, dtype=torch.float32)
    src = torch.empty(3, 0, dtype=torch.float32)
    index = torch.empty(3, 0, dtype=torch.long)

    ref = torch.scatter_reduce(
        inp.clone(), 1, index, src, reduce="prod", include_self=include_self
    )
    tri = scatter_reduce_(
        inp.clone(), 1, index, src, reduce="prod", include_self=include_self
    )

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("reduce", ["sum", "prod"])
def test_scatter_reduce_noncontiguous(reduce):
    torch.manual_seed(0)
    inp = torch.randn(4, 8, dtype=torch.float32).T
    src = torch.randn(4, 8, dtype=torch.float32).T
    index = torch.randint(0, inp.shape[0], src.shape, dtype=torch.long)

    ref = torch.scatter_reduce(inp.clone(), 0, index, src, reduce=reduce)
    tri = scatter_reduce_(inp.clone(), 0, index, src, reduce=reduce)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "dtype, reduce",
    [
        (torch.float16, "sum"),
        (torch.float64, "sum"),
        (torch.int32, "sum"),
        (torch.int32, "prod"),
        (torch.int32, "amin"),
    ],
)
def test_scatter_reduce_non_float32(dtype, reduce):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        inp = torch.randn(4, 8, dtype=dtype)
        src = torch.randn(4, 8, dtype=dtype)
    else:
        inp = torch.randint(-5, 6, (4, 8), dtype=dtype)
        src = torch.randint(-5, 6, (4, 8), dtype=dtype)
    index = torch.randint(0, 8, src.shape, dtype=torch.long)

    ref = torch.scatter_reduce(inp.clone(), 1, index, src, reduce=reduce)
    tri = scatter_reduce_(inp.clone(), 1, index, src, reduce=reduce)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [-3, 2])
def test_scatter_reduce_inplace_rejects_invalid_dim(dim):
    inp = torch.zeros((2, 2))
    index = torch.zeros((2, 2), dtype=torch.long)
    src = torch.ones((2, 2))

    with pytest.raises(IndexError):
        scatter_reduce_(inp, dim, index, src, reduce="sum")
