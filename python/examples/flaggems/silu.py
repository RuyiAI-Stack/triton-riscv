import torch
import triton
import triton.language as tl


@triton.jit
def silu_forward(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    out = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def silu_backward_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    dy = tl.load(dy_ptr + offsets, mask=mask)
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = 1.0 / (1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    tl.store(dx_ptr + offsets, dx, mask=mask)


def silu(x):
    A_c = x.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_forward[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x)


def silu_backward(grad_output, input):
    grad_input = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_backward_kernel[grid](
        input, grad_output, grad_input, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return grad_input


def silu_(A):
    result = silu(A)
    A.copy_(result)
    return A
