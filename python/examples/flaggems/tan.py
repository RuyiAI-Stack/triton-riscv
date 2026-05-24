import torch
import triton
import triton.language as tl


@triton.jit
def tan_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    y = tl.math.sin(xf) / tl.math.cos(xf)
    tl.store(y_ptr + offsets, y, mask=mask)


def tan(x):
    x_c = x.contiguous()
    n_elements = x_c.numel()
    y = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tan_kernel[grid](x_c, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y.view_as(x)


def tan_(x):
    x_c = x if x.is_contiguous() else x.contiguous()
    n_elements = x_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tan_kernel[grid](x_c, x_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    if not x.is_contiguous():
        x.copy_(x_c)
    return x
