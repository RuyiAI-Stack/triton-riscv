import torch
import triton
import triton.language as tl


@triton.jit
def ceil_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.ceil(x.to(tl.float32)).to(x.dtype), mask=mask)


def ceil(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    ceil_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def ceil_out(A, *, out=None):
    if out is None:
        return ceil(A)
    out.copy_(ceil(A))
    return out


def ceil_(A):
    result = ceil(A)
    A.copy_(result)
    return A
