import torch
import triton
import triton.language as tl


@triton.jit
def isneginf_kernel(
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
    res = xf == float("-inf")
    tl.store(out_ptr + offsets, res, mask=mask)


def isneginf(A):
    assert isinstance(A, torch.Tensor)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    out = torch.empty_like(A, dtype=torch.bool)
    A_c = A.contiguous()
    isneginf_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def isneginf_out(A, *, out=None):
    if out is None:
        return isneginf(A)
    result = isneginf(A)
    out.copy_(result)
    return out
