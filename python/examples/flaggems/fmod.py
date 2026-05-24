import torch
import triton
import triton.language as tl


@triton.jit
def _fmod(x, y):
    result = x / y
    quotient = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    return x - y * quotient


@triton.jit
def fmod_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    res = _fmod(x, y)

    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def fmod_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    res = _fmod(x, y_val)

    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def fmod_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)

    res = _fmod(x_val, y)

    tl.store(out_ptr + offsets, res, mask=mask)


def _invoke_fmod_kernel(A, B, out=None):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        common_dtype = torch.promote_types(A.dtype, B.dtype)
        A_c = A_c.to(common_dtype)
        B_c = B_c.to(common_dtype)
        if out is None:
            out = torch.empty_like(A_c, dtype=common_dtype)
        n_elements = A_c.numel()
        if n_elements == 0:
            return out.view_as(A)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fmod_kernel_tt[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        if out is None:
            out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        if n_elements == 0:
            return out.view_as(A)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fmod_kernel_ts[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        if out is None:
            out = torch.empty_like(B_c)
        n_elements = B_c.numel()
        if n_elements == 0:
            return out.view_as(B)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fmod_kernel_st[grid](A, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(B)
    else:
        return None


def fmod_tensor(A, B):
    res = _invoke_fmod_kernel(A, B)
    if res is not None:
        return res
    return torch.tensor(A % B)


def fmod_scalar(A, B):
    res = _invoke_fmod_kernel(A, B)
    if res is not None:
        return res
    return torch.tensor(A % B)


def fmod_tensor_(A, B):
    _invoke_fmod_kernel(A, B, out=A)
    return A


def fmod_scalar_(A, B):
    _invoke_fmod_kernel(A, B, out=A)
    return A
