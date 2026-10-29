import pytest
import torch

from .scatter import scatter, scatter_


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_1d(size, dim, dtype):
    torch.manual_seed(0)
    inp = torch.zeros(size, dtype=dtype, device="cpu")
    index = torch.randperm(size, device="cpu")
    src = torch.randn(size, dtype=dtype, device="cpu")

    out_triton = scatter(inp, dim, index, src)
    out_torch = torch.scatter(inp, dim, index, src)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_2d(size, dtype):
    torch.manual_seed(0)
    inp = torch.zeros((size, size), dtype=dtype, device="cpu")
    index = torch.rand(size, size, device="cpu").argsort(dim=0)
    src = torch.randn((size, size), dtype=dtype, device="cpu")

    out_triton = scatter(inp, 0, index, src)
    out_torch = torch.scatter(inp, 0, index, src)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_inplace(size, dtype):
    torch.manual_seed(0)
    inp = torch.zeros(size, dtype=dtype, device="cpu")
    index = torch.randperm(size, device="cpu")
    src = torch.randn(size, dtype=dtype, device="cpu")

    inp_clone = inp.clone()
    scatter_(inp, 0, index, src)
    inp_clone.scatter_(0, index, src)

    torch.testing.assert_close(inp, inp_clone, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("inplace", [False, True])
def test_scatter_duplicate_indices(size, inplace):
    inp = torch.full((size,), -1.0)
    index = torch.arange(size) % 17
    src = torch.arange(size, dtype=torch.float32)
    actual = scatter_(inp, 0, index, src) if inplace else scatter(inp, 0, index, src)

    winners = actual[:17]
    positions = winners.to(torch.int64)
    assert torch.all((positions >= 0) & (positions < size))
    torch.testing.assert_close(winners, src[positions], rtol=0, atol=0)
    torch.testing.assert_close(index[positions], torch.arange(17), rtol=0, atol=0)
    torch.testing.assert_close(actual[17:], torch.full_like(actual[17:], -1))
    if inplace:
        assert actual.data_ptr() == inp.data_ptr()
    else:
        torch.testing.assert_close(inp, torch.full_like(inp, -1))


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
