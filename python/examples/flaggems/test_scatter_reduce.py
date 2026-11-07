import pytest
import torch

from .scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out


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


@pytest.mark.parametrize("shape, dim", [((1, 4), 1), ((1, 1, 4), 2)])
def test_scatter_reduce_prod_special_values(shape, dim):
    inp = torch.tensor([float("nan"), float("inf"), 0.0, -0.0]).reshape(shape)
    src = torch.tensor([2.0, 0.0, -1.0, -1.0]).reshape(shape)
    index = torch.arange(4).reshape(shape)

    ref = torch.scatter_reduce(inp.clone(), dim, index, src, reduce="prod")
    tri = scatter_reduce(inp.clone(), dim, index, src, reduce="prod")

    torch.testing.assert_close(tri, ref, equal_nan=True)
    assert torch.equal(torch.signbit(tri), torch.signbit(ref))


@pytest.mark.parametrize("implementation", [scatter_reduce, scatter_reduce_])
@pytest.mark.parametrize("reduce", ["amax", "amin"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "inp_value, src_value",
    [(-0.0, 0.0), (0.0, -0.0), (float("nan"), 1.0), (1.0, float("nan"))],
)
def test_scatter_reduce_minmax_special_values(
    implementation, reduce, dtype, inp_value, src_value
):
    inp = torch.tensor([inp_value], dtype=dtype)
    src = torch.tensor([src_value], dtype=dtype)
    index = torch.zeros(1, dtype=torch.long)

    ref = torch.scatter_reduce(inp.clone(), 0, index, src, reduce=reduce)
    tri = implementation(inp.clone(), 0, index, src, reduce=reduce)

    torch.testing.assert_close(tri, ref, equal_nan=True)
    if not torch.isnan(ref).any():
        assert torch.equal(torch.signbit(tri), torch.signbit(ref))


@pytest.mark.parametrize("dim", [-3, 2])
def test_scatter_reduce_rejects_invalid_dim(dim):
    inp = torch.zeros((2, 2))
    index = torch.zeros((2, 2), dtype=torch.long)
    src = torch.ones((2, 2))

    with pytest.raises(IndexError):
        scatter_reduce(inp, dim, index, src, reduce="sum")


def test_scatter_reduce_out():
    torch.manual_seed(0)
    inp = torch.randn(4, 8, dtype=torch.float32)
    src = torch.randn(4, 8, dtype=torch.float32)
    index = torch.randint(0, 8, (4, 8), dtype=torch.long)
    ref = torch.scatter_reduce(inp, 1, index, src, reduce="sum")
    out = torch.empty_like(inp)

    returned = scatter_reduce_out(inp, 1, index, src, "sum", out=out)

    assert returned is out
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
