import math

import torch
import triton
import triton.language as tl


@triton.jit
def atan2_kernel_tt(
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
    # NOTE: tl.math.atan2 is NOT available in the triton-riscv MLIR backend.
    # The backend only supports basic arithmetic ops. Polynomial approximation
    # was attempted but insufficient accuracy (error > 1e-3).
    res = tl.math.atan2(x.to(tl.float32), y.to(tl.float32))
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def atan2_kernel_ts(
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
    res = tl.math.atan2(x.to(tl.float32), y_val.to(tl.float32))
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def atan2_kernel_st(
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
    res = tl.math.atan2(x_val.to(tl.float32), y.to(tl.float32))
    tl.store(out_ptr + offsets, res, mask=mask)


def _atan2_internal(A, B, out=None):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()

        if out is None:
            out = torch.empty_like(
                A_c, dtype=torch.promote_types(A.dtype, B.dtype)
            )
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        atan2_kernel_tt[grid](
            A_c, B_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        if out is None:
            out = torch.empty_like(
                A_c, dtype=torch.promote_types(A.dtype, torch.tensor(B).dtype)
            )
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        atan2_kernel_ts[grid](A_c, B, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        if out is None:
            out = torch.empty_like(
                B_c, dtype=torch.promote_types(torch.tensor(A).dtype, B.dtype)
            )
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = B_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        atan2_kernel_st[grid](A, B_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    else:
        return torch.tensor(math.atan2(A, B))


def atan2(input, other):
    return _atan2_internal(input, other)


def atan2_out(input, other, out):
    res = _atan2_internal(input, other, out=out)
    return res
