import pytest
import torch

from .scatter_add_ import scatter_add_


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_add_1d(size, dtype):
    torch.manual_seed(0)
    inp = torch.zeros(size, dtype=dtype, device="cpu")
    index = torch.randint(0, size, (size,), device="cpu")
    src = torch.randn(size, dtype=dtype, device="cpu")

    inp_triton = inp.clone()
    inp_torch = inp.clone()

    scatter_add_(inp_triton, 0, index, src)
    inp_torch.scatter_add_(0, index, src)

    torch.testing.assert_close(inp_triton, inp_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_add_2d(size, dtype):
    torch.manual_seed(0)
    inp = torch.zeros((size, size), dtype=dtype, device="cpu")
    index = torch.randint(0, size, (size, size), device="cpu")
    src = torch.randn((size, size), dtype=dtype, device="cpu")

    inp_triton = inp.clone()
    inp_torch = inp.clone()

    scatter_add_(inp_triton, 0, index, src)
    inp_torch.scatter_add_(0, index, src)

    torch.testing.assert_close(inp_triton, inp_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_add_dim1(size, dtype):
    torch.manual_seed(0)
    inp = torch.zeros((size, size), dtype=dtype, device="cpu")
    index = torch.randint(0, size, (size, size), device="cpu")
    src = torch.randn((size, size), dtype=dtype, device="cpu")

    inp_triton = inp.clone()
    inp_torch = inp.clone()

    scatter_add_(inp_triton, 1, index, src)
    inp_torch.scatter_add_(1, index, src)

    torch.testing.assert_close(inp_triton, inp_torch, rtol=1e-3, atol=1e-3)
