import pytest
import torch

from .triu import triu, triu_, triu_out


@pytest.mark.parametrize("shape", [(32, 32), (16, 64, 64), (1024, 1024)])
@pytest.mark.parametrize("diagonal", [0, 1, -1, 5, -5])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_triu_forward(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cpu")

    out_triton = triu(inp, diagonal)
    out_torch = torch.triu(inp, diagonal)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(32, 32), (16, 64, 64)])
@pytest.mark.parametrize("diagonal", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_triu_inplace(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cpu")
    inp_clone = inp.clone()

    out_triton = triu_(inp, diagonal)
    out_torch = inp_clone.triu_(diagonal)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(inp, inp_clone, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(32, 32), (1024, 1024)])
@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_triu_out(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cpu")
    out = torch.empty(shape, dtype=dtype, device="cpu")

    ref = torch.triu(inp, diagonal)
    tri = triu_out(inp, diagonal, out=out)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
