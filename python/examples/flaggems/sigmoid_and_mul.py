import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_and_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x_f32))
    tl.store(out_ptr + offsets, sigmoid * y_f32, mask=mask)


@triton.jit
def sigmoid_and_mul_backward_kernel(
    grad_ptr,
    x_ptr,
    y_ptr,
    dx_ptr,
    dy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    dx = grad * y * sigmoid * (1.0 - sigmoid)
    dy = grad * sigmoid
    tl.store(dx_ptr + offsets, dx, mask=mask)
    tl.store(dy_ptr + offsets, dy, mask=mask)


def sigmoid_and_mul(x, y):
    x, y = torch.broadcast_tensors(x, y)
    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_and_mul_kernel[grid](
        x_c, y_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view_as(x)


def sigmoid_and_mul_backward(grad_output, x, y):
    x, y, grad_output = torch.broadcast_tensors(x, y, grad_output)
    x_c = x.contiguous()
    y_c = y.contiguous()
    grad_c = grad_output.contiguous()
    dx = torch.empty_like(x_c)
    dy = torch.empty_like(y_c)
    n_elements = x_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_and_mul_backward_kernel[grid](
        grad_c, x_c, y_c, dx, dy, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return dx.view_as(x), dy.view_as(y)
