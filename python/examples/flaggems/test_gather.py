import pytest
import torch

from .gather import gather


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_gather_forward_1d(shape):
    inp = torch.randn(shape, device="cpu", dtype=torch.float32)
    dim = 0
    index = torch.randint(
        0, shape[dim], shape, device="cpu", dtype=torch.int64
    )
    ref = torch.gather(inp, dim, index)
    out = gather(inp, dim, index)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(128, 512), (128, 1023), (128, 1024)])
def test_gather_forward_2d(shape):
    inp = torch.randn(shape, device="cpu", dtype=torch.float32)
    for dim in [0, 1]:
        index = torch.randint(
            0, shape[dim], shape, device="cpu", dtype=torch.int64
        )
        ref = torch.gather(inp, dim, index)
        out = gather(inp, dim, index)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_gather_out_of_place():
    inp = torch.randn((64, 512), device="cpu", dtype=torch.float32)
    dim = 0
    index = torch.randint(0, 64, (32, 512), device="cpu", dtype=torch.int64)
    out = gather(inp, dim, index)
    ref = torch.gather(inp, dim, index)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert out.data_ptr() != inp.data_ptr()


def test_gather_negative_index():
    shape = (128, 512)
    inp = torch.randn(shape, device="cpu", dtype=torch.float32)
    dim = 1
    # Negative indices must be wrapped to positive
    index = torch.full((128, 256), -1, device="cpu", dtype=torch.int64)
    index = index % shape[dim]
    ref = torch.gather(inp, dim, index)
    out = gather(inp, dim, index)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(128, 512), (64, 256)])
def test_gather_backward(shape):
    torch.manual_seed(0)
    inp = torch.randn(
        shape, device="cpu", dtype=torch.float32, requires_grad=True
    )
    dim = 0
    index = torch.randint(
        0, shape[dim], shape, device="cpu", dtype=torch.int64
    )

    # Reference backward via torch
    ref_out = torch.gather(inp, dim, index)
    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    ref_dx = inp.grad.clone()

    # Triton backward
    from .gather import gather_backward

    inp2 = inp.detach().clone().requires_grad_(True)
    tri_dx = gather_backward(grad, inp2, dim, index, sparse_grad=False)

    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-4, atol=1e-4)
