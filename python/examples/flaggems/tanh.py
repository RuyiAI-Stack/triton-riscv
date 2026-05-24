import torch
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    exp2x = tl.exp(2.0 * x_f32)
    tl.store(out_ptr + offsets, (exp2x - 1.0) / (exp2x + 1.0), mask=mask)


@triton.jit
def tanh_grad_kernel(
    y_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    dy = tl.load(dy_ptr + offsets, mask=mask)
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    tl.store(dx_ptr + offsets, dy_f32 * (1.0 - y_f32 * y_f32), mask=mask)


def tanh(x):
    A_c = x.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tanh_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x)


def tanh_backward(grad_output, output):
    y = output.contiguous()
    dy = grad_output.contiguous()
    dx = torch.empty_like(y)
    n_elements = y.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tanh_grad_kernel[grid](y, dy, dx, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return dx


def tanh_(A):
    result = tanh(A)
    A.copy_(result)
    return A
