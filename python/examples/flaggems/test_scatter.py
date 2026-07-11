import pytest
import torch

from .scatter import scatter, scatter_


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_1d(size, dim, dtype):
    torch.manual_seed(0)
    inp = torch.zeros(size, dtype=dtype, device="cpu")
    index = torch.randint(0, size, (size,), device="cpu")
    src = torch.randn(size, dtype=dtype, device="cpu")

    out_triton = scatter(inp, dim, index, src)
    out_torch = torch.scatter(inp, dim, index, src)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_2d(size, dtype):
    torch.manual_seed(0)
    inp = torch.zeros((size, size), dtype=dtype, device="cpu")
    index = torch.randint(0, size, (size, size), device="cpu")
    src = torch.randn((size, size), dtype=dtype, device="cpu")

    out_triton = scatter(inp, 0, index, src)
    out_torch = torch.scatter(inp, 0, index, src)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_inplace(size, dtype):
    torch.manual_seed(0)
    inp = torch.zeros(size, dtype=dtype, device="cpu")
    index = torch.randint(0, size, (size,), device="cpu")
    src = torch.randn(size, dtype=dtype, device="cpu")

    inp_clone = inp.clone()
    scatter_(inp, 0, index, src)
    inp_clone.scatter_(0, index, src)

    torch.testing.assert_close(inp, inp_clone, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_add_reduce(size, dtype):
    torch.manual_seed(0)
    inp = torch.zeros(size, dtype=dtype, device="cpu")
    index = torch.randint(0, size, (size,), device="cpu")
    src = torch.randn(size, dtype=dtype, device="cpu")

    out_triton = scatter(inp, 0, index, src, reduce="add")
    out_torch = torch.scatter_add(inp, 0, index, src)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reduce", [None, "add"])
def test_scatter_partial_index_and_larger_src(dim, reduce):
    torch.manual_seed(0)
    inp = torch.randn((3, 5), dtype=torch.float32)
    index = torch.empty((2, 4), dtype=torch.int64)
    index.random_(0, inp.shape[dim])
    # A larger src has different contiguous strides from index; only its
    # leading index-shaped region participates in scatter.
    src = torch.randn((3, 5), dtype=torch.float32)

    tri = scatter(inp, dim, index, src, reduce=reduce)
    if reduce == "add":
        ref = torch.scatter_add(inp, dim, index, src)
    else:
        ref = torch.scatter(inp, dim, index, src)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)
