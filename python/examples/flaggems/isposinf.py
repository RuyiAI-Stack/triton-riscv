import torch
import triton
import triton.language as tl


@triton.jit
def isposinf_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    res = xf == float("inf")
    tl.store(out_ptr + offsets, res, mask=mask)


def isposinf(A):
    assert isinstance(A, torch.Tensor)
    A_c = A.contiguous()
    out = torch.empty_like(A_c, dtype=torch.bool)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    isposinf_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def isposinf_out(A, *, out=None):
    if out is None:
        return isposinf(A)
    result = isposinf(A)
    out.copy_(result)
    return out
