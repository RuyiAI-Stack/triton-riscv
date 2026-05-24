import torch
import triton
import triton.language as tl


@triton.jit
def rsub_kernel_tt(
    x_ptr,
    y_ptr,
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
    y = tl.load(y_ptr + offsets, mask=mask)
    res = y - x * alpha
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def rsub_kernel_ts(
    x_ptr,
    y_val,
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
    res = y_val - x * alpha
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def rsub_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    res = y - x_val * alpha
    tl.store(out_ptr + offsets, res, mask=mask)


def _rsub_internal(A, B, alpha=1):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        out_dtype = torch.result_type(A_c, B_c)
        out = torch.empty_like(A_c, dtype=out_dtype)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        rsub_kernel_tt[grid](
            A_c, B_c, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE
        )
        return out.view_as(A)
    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        rsub_kernel_ts[grid](
            A_c, B, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE
        )
        return out.view_as(A)
    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        out = torch.empty_like(B_c)
        n_elements = B_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        rsub_kernel_st[grid](
            A, B_c, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE
        )
        return out.view_as(B)
    else:
        return torch.tensor(B - A * alpha)


def rsub(A, B, *, alpha=1):
    return _rsub_internal(A, B, alpha)


def rsub_tensor(A, B, *, alpha=1):
    return rsub(A, B, alpha=alpha)


def rsub_scalar(A, B, alpha=1):
    return rsub(A, B, alpha=alpha)
