import torch
import triton
import triton.language as tl


@triton.jit
def cosh_kernel(
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
    out = 0.5 * (tl.exp(x_fp32) + tl.exp(-x_fp32))
    tl.store(out_ptr + offsets, out, mask=mask)


def cosh(A):
    if not A.dtype.is_floating_point:
        result_dtype = torch.float32
    else:
        result_dtype = A.dtype
    A_c = A.contiguous()
    out = torch.empty_like(A_c, dtype=result_dtype)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    cosh_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def cosh_out(A, *, out=None):
    if out is None:
        return cosh(A)
    out.copy_(cosh(A))
    return out


def cosh_(A):
    result = cosh(A)
    A.copy_(result)
    return A
