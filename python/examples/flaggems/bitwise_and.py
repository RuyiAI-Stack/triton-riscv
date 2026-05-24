import torch
import triton
import triton.language as tl


@triton.jit
def bitwise_and_kernel_tt(
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
    tl.store(out_ptr + offsets, x & y, mask=mask)


@triton.jit
def bitwise_and_kernel_ts(
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
    tl.store(out_ptr + offsets, x & y_val, mask=mask)


@triton.jit
def bitwise_and_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x_val & y, mask=mask)


def _bitwise_and_impl(A, B, out=None):
    BLOCK_SIZE = 1024
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        if out is None:
            out = torch.empty_like(A_c)
        out_c = out if out.is_contiguous() else out.contiguous()
        n_elements = A_c.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        bitwise_and_kernel_tt[grid](
            A_c, B_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        if out is None:
            out = torch.empty_like(A_c)
        out_c = out if out.is_contiguous() else out.contiguous()
        n_elements = A_c.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        bitwise_and_kernel_ts[grid](
            A_c, B, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        if out is None:
            out = torch.empty_like(B_c)
        out_c = out if out.is_contiguous() else out.contiguous()
        n_elements = B_c.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        bitwise_and_kernel_st[grid](
            A, B_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    else:
        return torch.tensor(A & B)


def bitwise_and_tensor(A, B):
    return _bitwise_and_impl(A, B)


def bitwise_and_tensor_(A, B):
    return _bitwise_and_impl(A, B, out=A)


def bitwise_and_scalar(A, B):
    return _bitwise_and_impl(A, B)


def bitwise_and_scalar_(A, B):
    return _bitwise_and_impl(A, B, out=A)


def bitwise_and_scalar_tensor(A, B):
    return _bitwise_and_impl(B, A)
