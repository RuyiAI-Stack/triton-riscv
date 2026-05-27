import torch
import triton
import triton.language as tl


@triton.jit
def exp2_kernel(
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
    tl.store(out_ptr + offsets, tl.exp2(x_f32), mask=mask)


@triton.jit
def exp2_grad_kernel(
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
    tl.store(dx_ptr + offsets, dy_f32 * y_f32 * 0.6931471805599453, mask=mask)


def exp2(x):
    A_c = x.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    exp2_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x)


def exp2_(x):
    result = exp2(x)
    x.copy_(result)
    return x
