import torch
import triton
import triton.language as tl


@triton.jit
def minimum_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if x.dtype == tl.bfloat16:
        x = x.to(tl.float32)
        y = y.to(tl.float32)
    res = tl.minimum(x, y)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def minimum_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if x.dtype == tl.bfloat16:
        x = x.to(tl.float32)
        y_val = y_val.to(tl.float32)
    res = tl.minimum(x, y_val)
    tl.store(out_ptr + offsets, res, mask=mask)


def _minimum_internal(A, B):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        minimum_kernel_tt[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        minimum_kernel_ts[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        out = torch.empty_like(B_c)
        n_elements = B_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        minimum_kernel_ts[grid](B_c, A, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(B)
    else:
        return torch.tensor(min(A, B))


def minimum(A, B):
    return _minimum_internal(A, B)
