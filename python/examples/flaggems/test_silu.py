import pytest
import torch

from .silu import silu, silu_, silu_backward


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_silu(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.silu(x)
    tri = silu(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_silu_fp16(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float16, device="cpu")

    ref = torch.nn.functional.silu(x)
    tri = silu(x)

    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


def test_silu_inplace():
    x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    x_ref = torch.nn.functional.silu(x_ref)
    silu_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_silu_backward(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu", requires_grad=True)

    out = torch.nn.functional.silu(x)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    ref_grad = x.grad.clone()

    x.grad.zero_()
    tri_grad = silu_backward(grad_output, x)

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-3, atol=1e-3)
