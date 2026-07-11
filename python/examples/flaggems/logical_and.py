import torch
import triton
import triton.language as tl


@triton.jit
def logical_and_kernel(
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
    res = x & y
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def logical_and_kernel_(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    res = tl.where((x != 0) & (y != 0), 1, 0)
    tl.store(x_ptr + offsets, res, mask=mask)


def logical_and(A, B):
    A, B = torch.broadcast_tensors(A, B)
    result_dtype = torch.result_type(A, B)
    A_c = A.to(torch.uint8) if A.dtype == torch.bool else A.contiguous()
    B_c = B.to(torch.uint8) if B.dtype == torch.bool else B.contiguous()
    kernel_dtype = torch.uint8 if result_dtype == torch.bool else result_dtype
    out = torch.empty_like(A_c, dtype=kernel_dtype)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    logical_and_kernel[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    result = out.bool() if result_dtype == torch.bool else out
    return result.view_as(A)


def logical_and_(A, B):
    A.copy_(logical_and(A, B))
    return A
