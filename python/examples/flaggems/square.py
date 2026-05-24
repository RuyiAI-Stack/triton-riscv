import torch
import triton
import triton.language as tl


@triton.jit
def square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * x, mask=mask)


def square(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    square_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def square_out(A, *, out=None):
    if out is None:
        return square(A)
    A_c = A.contiguous()
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    square_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def square_(A):
    result = square(A)
    A.copy_(result)
    return A
