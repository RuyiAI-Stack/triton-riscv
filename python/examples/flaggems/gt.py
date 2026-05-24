import torch
import triton
import triton.language as tl


@triton.jit
def gt_kernel_tt(
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
    res = x.to(tl.float32) > y
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def gt_kernel_ts(
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
    res = x.to(tl.float32) > y_val
    tl.store(out_ptr + offsets, res, mask=mask)


def gt(A, B):
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    out = torch.empty_like(A, dtype=torch.bool)
    A_c = A.contiguous()
    B_c = B.contiguous()
    gt_kernel_tt[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def gt_scalar(A, B):
    assert isinstance(A, torch.Tensor)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    out = torch.empty_like(A, dtype=torch.bool)
    A_c = A.contiguous()
    gt_kernel_ts[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
