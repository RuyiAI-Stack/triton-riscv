import torch
import triton
import triton.language as tl


@triton.jit
def sub_kernel_tt(
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
    res = x - y * alpha
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def sub_kernel_ts(
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
    res = x - y_val * alpha
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def sub_kernel_st(
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
    res = x_val - y * alpha
    tl.store(out_ptr + offsets, res, mask=mask)


def _sub_internal(A, B, alpha=1):
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
        sub_kernel_tt[grid](A_c, B_c, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub_kernel_ts[grid](A_c, B, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        out = torch.empty_like(B_c)
        n_elements = B_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub_kernel_st[grid](A, B_c, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(B)
    else:
        return torch.tensor(A - B * alpha)


def sub(A, B, *, alpha=1):
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        if A_is_complex and B_is_complex:
            Ar = torch.view_as_real(A)
            Br = torch.view_as_real(B)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = _sub_internal(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        elif A_is_complex and not B_is_complex:
            Ar = torch.view_as_real(A)
            if isinstance(B, torch.Tensor):
                Br = torch.view_as_real(B.to(A.dtype))
            else:
                Br = torch.view_as_real(
                    torch.tensor(B, dtype=A.dtype, device=A.device).expand_as(A)
                )
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = _sub_internal(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        else:
            Br = torch.view_as_real(B)
            if isinstance(A, torch.Tensor):
                Ar = torch.view_as_real(A.to(B.dtype))
            else:
                Ar = torch.view_as_real(
                    torch.tensor(A, dtype=B.dtype, device=B.device).expand_as(B)
                )
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = _sub_internal(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))

    return _sub_internal(A, B, alpha)


def sub_(A, B, *, alpha=1):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub_kernel_tt[grid](A_c, B_c, A_c, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
        if A_c.data_ptr() != A.data_ptr():
            A.copy_(A_c)
        return A
    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub_kernel_ts[grid](A_c, B, A_c, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
        if A_c.data_ptr() != A.data_ptr():
            A.copy_(A_c)
        return A
    else:
        raise TypeError("sub_ requires A to be a Tensor")
