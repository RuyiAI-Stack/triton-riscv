import torch
import triton
import triton.language as tl


@triton.jit
def rsqrt_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, 1.0 / tl.sqrt(x.to(tl.float32)), mask=mask)


def rsqrt(A):
    A_c = A.contiguous()
    out_dtype = (
        torch.float32
        if not A_c.is_floating_point() and not A_c.is_complex()
        else A_c.dtype
    )
    out = torch.empty_like(A_c, dtype=out_dtype)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rsqrt_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def rsqrt_(A):
    result = rsqrt(A)
    A.copy_(result)
    return A
