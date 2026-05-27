import torch
import triton
import triton.language as tl


@triton.jit
def sin_kernel(
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
    y = tl.sin(x.to(tl.float32))
    tl.store(y_ptr + offsets, y, mask=mask)


def sin(A):
    A_c = A.contiguous()
    n_elements = A_c.numel()
    y = torch.empty_like(A_c)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sin_kernel[grid](A_c, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y.view_as(A)


def sin_(A):
    A_c = A if A.is_contiguous() else A.contiguous()
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sin_kernel[grid](A_c, A_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    if not A.is_contiguous():
        A.copy_(A_c)
    return A
