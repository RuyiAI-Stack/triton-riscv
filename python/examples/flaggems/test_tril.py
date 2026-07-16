import pytest
import torch

from .tril import tril, tril_, tril_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_tril_forward(size, diagonal, dtype):
    inp = torch.randn((size, size), dtype=dtype, device="cpu")

    out_triton = tril(inp, diagonal)
    out_torch = torch.tril(inp, diagonal)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("diagonal", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_tril_inplace(size, diagonal, dtype):
    inp = torch.randn((size, size), dtype=dtype, device="cpu")
    inp_clone = inp.clone()

    out_triton = tril_(inp, diagonal)
    out_torch = torch.tril(inp_clone, diagonal, out=inp_clone)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(inp, inp_clone, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1024])
@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_tril_out(size, diagonal, dtype):
    inp = torch.randn((size, size), dtype=dtype, device="cpu")
    out = torch.empty((size, size), dtype=dtype, device="cpu")

    ref = torch.tril(inp, diagonal)
    tri = tril_out(inp, diagonal, out=out)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
