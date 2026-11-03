import torch
import triton
import triton.language as tl


@triton.jit
def tanh_and_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tanh_x = 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0

    tl.store(out_ptr + offsets, tanh_x * y, mask=mask)


@triton.jit
def tanh_and_mul_backward_kernel(
    grad_ptr,
    x_ptr,
    y_ptr,
    dx_ptr,
    dy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tanh_x = 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0

    dx = grad * y * (1.0 - tanh_x * tanh_x)
    dy = grad * tanh_x
    tl.store(dx_ptr + offsets, dx, mask=mask)
    tl.store(dy_ptr + offsets, dy, mask=mask)


def tanh_and_mul(x, y):
    x, y = torch.broadcast_tensors(x, y)
    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    tanh_and_mul_kernel[grid](
        x_c, y_c, out, n_elements, BLOCK_SIZE=block_size
    )
    return out.view_as(x)


def tanh_and_mul_backward(grad_output, x, y):
    x_shape = x.shape
    y_shape = y.shape
    x, y, grad_output = torch.broadcast_tensors(x, y, grad_output)
    x_c = x.contiguous()
    y_c = y.contiguous()
    grad_c = grad_output.contiguous()
    dx = torch.empty_like(x_c)
    dy = torch.empty_like(y_c)
    n_elements = x_c.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    tanh_and_mul_backward_kernel[grid](
        grad_c,
        x_c,
        y_c,
        dx,
        dy,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return dx.sum_to_size(x_shape), dy.sum_to_size(y_shape)
