import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["alpha"])
def celu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    result = tl.where(x_f32 > 0, x_f32, alpha * (tl.exp(x_f32 / alpha) - 1.0))
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


def celu(A, alpha=1.0):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    celu_kernel[grid](A_c, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def celu_(A, alpha=1.0):
    result = celu(A, alpha=alpha)
    A.copy_(result)
    return A
