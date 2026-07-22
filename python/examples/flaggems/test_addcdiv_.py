import pytest
import torch

from .addcdiv_ import addcdiv_


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_addcdiv_(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    t1 = torch.randn(shape, dtype=dtype, device="cpu")
    t2 = torch.randn(shape, dtype=dtype, device="cpu") + 2.0
    x_ref = x.clone()

    torch.addcdiv(x_ref, t1, t2, value=0.5, out=x_ref)
    addcdiv_(x, t1, t2, value=0.5)

    tri_out = x
    ref_out = x_ref

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(16, 16), (8, 4, 2)])
@pytest.mark.parametrize("value", [0.25, 2.5])
def test_addcdiv___broadcast(dtype, shape, value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    # broadcastable input and divisor
    t1 = torch.randn((1,) + shape[1:], dtype=dtype, device="cpu")
    t2 = torch.randn((shape[0], 1) + shape[2:], dtype=dtype, device="cpu")

    x = torch.where(torch.abs(x) < 1.0, torch.sign(x) + 1.0, x)
    t2 = torch.where(torch.abs(t2) < 0.1, torch.sign(t2) * 0.1 + 0.1, t2)

    ref_out = torch.addcdiv(x.clone(), t1, t2, value=value)
    tri_out = addcdiv_(x, t1, t2, value=value)

    assert tri_out is x
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(4, 5, 3)])
def test_addcdiv__non_contiguous(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cpu").transpose(1, 2)
    x_nc = x.transpose(0, 1)  # non-contiguous view
    t1 = torch.randn((3, 1, 5), dtype=dtype, device="cpu")
    t2 = torch.randn((3, 1, 5), dtype=dtype, device="cpu")
    t2 = torch.where(torch.abs(t2) < 0.1, torch.sign(t2) * 0.1 + 0.1, t2)

    ref_out = torch.addcdiv(x_nc.clone(), t1, t2)
    tri_out = addcdiv_(x_nc, t1, t2)
    assert tri_out is x_nc
    torch.testing.assert_close(tri_out, ref_out)
