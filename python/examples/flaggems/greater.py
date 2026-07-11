import torch
import triton
import triton.language as tl


@triton.jit
def greater_kernel_tt(
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
    res = x > y
    tl.store(out_ptr + offsets, res.to(tl.int8), mask=mask)


@triton.jit
def greater_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    res = x > y_val
    tl.store(out_ptr + offsets, res.to(tl.int8), mask=mask)


def greater(A, B):
    A, B = torch.broadcast_tensors(A, B)
    A_c = A.contiguous()
    B_c = B.contiguous()
    out = torch.empty_like(A_c, dtype=torch.uint8)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    greater_kernel_tt[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.bool().view_as(A)


def greater_out(A, B, *, out=None):
    if out is None:
        return greater(A, B)
    A, B = torch.broadcast_tensors(A, B)
    A_c = A.contiguous()
    B_c = B.contiguous()
    n_elements = A_c.numel()
    kernel_out = torch.empty_like(A_c, dtype=torch.uint8)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    greater_kernel_tt[grid](
        A_c, B_c, kernel_out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    out.copy_(kernel_out.bool())
    return out


def greater_scalar(A, B):
    A_c = A.contiguous()
    out = torch.empty_like(A_c, dtype=torch.uint8)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    greater_kernel_ts[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.bool().view_as(A)


def greater_scalar_out(A, B, *, out=None):
    if out is None:
        return greater_scalar(A, B)
    A_c = A.contiguous()
    n_elements = A_c.numel()
    kernel_out = torch.empty_like(A_c, dtype=torch.uint8)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    greater_kernel_ts[grid](
        A_c, B, kernel_out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    out.copy_(kernel_out.bool())
    return out
