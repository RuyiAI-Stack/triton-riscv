import torch
import triton
import triton.language as tl


@triton.jit
def expm1_kernel(
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
    tl.store(out_ptr + offsets, tl.exp(x_f32) - 1.0, mask=mask)


@triton.jit
def expm1_grad_kernel(
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
    x_f32 = x.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    tl.store(dx_ptr + offsets, dy_f32 * tl.exp(x_f32), mask=mask)


def expm1(x):
    A_c = x.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    expm1_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x)


def expm1_(x):
    result = expm1(x)
    x.copy_(result)
    return x


def expm1_out(x, out):
    result = expm1(x)
    out.copy_(result)
    return out
