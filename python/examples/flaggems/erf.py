import torch
import triton
import triton.language as tl


@triton.jit
def erf_kernel(
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
    tl.store(out_ptr + offsets, tl.math.erf(x_f32), mask=mask)


def erf(x):
    A_c = x.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    erf_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x)


def erf_(x):
    result = erf(x)
    x.copy_(result)
    return x
