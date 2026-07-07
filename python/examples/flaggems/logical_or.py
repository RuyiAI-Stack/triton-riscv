import torch
import triton
import triton.language as tl


@triton.jit
def logical_or_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    res = (x != 0) | (y != 0)
    tl.store(out_ptr + offsets, res.to(tl.uint8), mask=mask)


def logical_or(A, B):
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    out = torch.empty_like(A, dtype=torch.uint8)
    A_c = A.to(torch.uint8).contiguous() if A.dtype == torch.bool else A.contiguous()
    B_c = B.to(torch.uint8).contiguous() if B.dtype == torch.bool else B.contiguous()
    logical_or_kernel[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.to(torch.bool)


def logical_or_(A, B):
    result = logical_or(A, B)
    A.copy_(result)
    return A
