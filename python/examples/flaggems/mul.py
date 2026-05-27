import torch
import triton
import triton.language as tl


@triton.jit
def mul_kernel_tt(
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
    res = x * y
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def mul_kernel_ts(
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
    res = x * y_val
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def mul_kernel_st(
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
    res = x_val * y
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def mul_kernel_complex(
    ar_ptr,
    ai_ptr,
    br_ptr,
    bi_ptr,
    out_r_ptr,
    out_i_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    ar = tl.load(ar_ptr + offsets, mask=mask)
    ai = tl.load(ai_ptr + offsets, mask=mask)
    br = tl.load(br_ptr + offsets, mask=mask)
    bi = tl.load(bi_ptr + offsets, mask=mask)
    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    tl.store(out_r_ptr + offsets, real, mask=mask)
    tl.store(out_i_ptr + offsets, imag, mask=mask)


def mul(A, B):
    if isinstance(A, torch.Tensor) and A.is_complex():
        if isinstance(B, torch.Tensor) and B.is_complex():
            A_r = A.real.contiguous()
            A_i = A.imag.contiguous()
            B_r = B.real.contiguous()
            B_i = B.imag.contiguous()
            A_r, B_r = torch.broadcast_tensors(A_r, B_r)
            A_i, B_i = torch.broadcast_tensors(A_i, B_i)
            out_r = torch.empty_like(A_r)
            out_i = torch.empty_like(A_i)
            n_elements = A_r.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            mul_kernel_complex[grid](
                A_r,
                A_i,
                B_r,
                B_i,
                out_r,
                out_i,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            out = torch.complex(out_r, out_i)
            return out.view_as(A_r)
        else:
            return torch.mul(A, B)
    elif isinstance(B, torch.Tensor) and B.is_complex():
        return torch.mul(A, B)

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
        mul_kernel_tt[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        mul_kernel_ts[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        out = torch.empty_like(B_c)
        n_elements = B_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        mul_kernel_st[grid](A, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(B)
    else:
        return torch.tensor(A * B)


def mul_(A, B):
    res = mul(A, B)
    A.copy_(res)
    return A
