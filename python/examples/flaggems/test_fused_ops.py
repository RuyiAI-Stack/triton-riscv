import pytest
import torch
import triton

from .gelu_and_mul import (
    gelu_and_mul,
    gelu_none_and_mul_grad_kernel,
    gelu_none_and_mul_kernel,
    gelu_tanh_and_mul_grad_kernel,
    gelu_tanh_and_mul_kernel,
)
from .silu_and_mul import (
    silu_and_mul,
    silu_and_mul_grad_kernel,
    silu_and_mul_kernel,
    silu_and_mul_out,
)


def launch_kernel(
    kernel, x, y, out=None, is_grad=False, dgrad=None, dx=None, dy=None
):
    n_elements = x.numel()
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    if not is_grad:
        if out is None:
            out = torch.empty_like(x)
        kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out
    kernel[grid](x, y, dgrad, dx, dy, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return dx, dy


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_silu_and_mul(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.nn.functional.silu(x) * y

    tri_out = launch_kernel(silu_and_mul_kernel, x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_silu_and_mul_autograd(size):
    torch.manual_seed(0)
    x = torch.randn(
        size, device="cpu", dtype=torch.float32, requires_grad=True
    )
    y = torch.randn(
        size, device="cpu", dtype=torch.float32, requires_grad=True
    )
    dgrad = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.nn.functional.silu(x) * y
    tri_out = silu_and_mul(x, y)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)

    ref_out.backward(dgrad)
    ref_dx, ref_dy = x.grad.clone(), y.grad.clone()

    x.grad, y.grad = None, None
    tri_out.backward(dgrad)
    tri_dx, tri_dy = x.grad, y.grad

    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_dy, ref_dy, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_silu_and_mul_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)
    out = torch.empty_like(x)
    ref_out = torch.nn.functional.silu(x) * y
    silu_and_mul_out(x, y, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_and_mul(size, approximate):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.nn.functional.gelu(x, approximate=approximate) * y

    if approximate == "none":
        tri_out = launch_kernel(gelu_none_and_mul_kernel, x, y)
    else:
        tri_out = launch_kernel(gelu_tanh_and_mul_kernel, x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_and_mul_autograd(size, approximate):
    torch.manual_seed(0)
    x = torch.randn(
        size, device="cpu", dtype=torch.float32, requires_grad=True
    )
    y = torch.randn(
        size, device="cpu", dtype=torch.float32, requires_grad=True
    )
    dgrad = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = torch.nn.functional.gelu(x, approximate=approximate) * y
    tri_out = gelu_and_mul(x, y, approximate=approximate)
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)

    ref_out.backward(dgrad)
    ref_dx, ref_dy = x.grad.clone(), y.grad.clone()

    x.grad, y.grad = None, None
    tri_out.backward(dgrad)
    tri_dx, tri_dy = x.grad, y.grad

    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_dy, ref_dy, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_silu_grad_coverage(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)
    dgrad = torch.randn(size, device="cpu", dtype=torch.float32)

    dx = torch.empty_like(x)
    dy = torch.empty_like(y)

    launch_kernel(
        silu_and_mul_grad_kernel,
        x,
        y,
        is_grad=True,
        dgrad=dgrad,
        dx=dx,
        dy=dy,
    )


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_grad_coverage(size, approximate):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)
    dgrad = torch.randn(size, device="cpu", dtype=torch.float32)

    dx = torch.empty_like(x)
    dy = torch.empty_like(y)

    kernel = (
        gelu_none_and_mul_grad_kernel
        if approximate == "none"
        else gelu_tanh_and_mul_grad_kernel
    )

    launch_kernel(kernel, x, y, is_grad=True, dgrad=dgrad, dx=dx, dy=dy)
