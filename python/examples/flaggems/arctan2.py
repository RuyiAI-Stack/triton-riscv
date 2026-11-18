import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def arctan2_kernel_tt(
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
    res = libdevice.atan2(x.to(tl.float32), y.to(tl.float32))
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def arctan2_kernel_ts(
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
    res = libdevice.atan2(x.to(tl.float32), y_val.to(tl.float32))
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def arctan2_kernel_st(
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
    res = libdevice.atan2(x_val.to(tl.float32), y.to(tl.float32))
    tl.store(out_ptr + offsets, res, mask=mask)


def _result_dtype(dtype_a, dtype_b):
    # Upstream declares the INT_TO_FLOAT promotion: integer inputs produce a
    # float32 result (matching torch's own type promotion), never an int out.
    out_dtype = torch.promote_types(dtype_a, dtype_b)
    if not out_dtype.is_floating_point:
        out_dtype = torch.float32
    return out_dtype


def _arctan2_internal(A, B, out=None):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()

        if out is None:
            out = torch.empty_like(A_c, dtype=_result_dtype(A.dtype, B.dtype))
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = A_c.numel()
        if n_elements == 0:
            if out_c.data_ptr() != out.data_ptr():
                out.copy_(out_c)
            return out

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        arctan2_kernel_tt[grid](A_c, B_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        if out is None:
            out = torch.empty_like(
                A_c, dtype=_result_dtype(A.dtype, torch.tensor(B).dtype)
            )
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = A_c.numel()
        if n_elements == 0:
            if out_c.data_ptr() != out.data_ptr():
                out.copy_(out_c)
            return out

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        arctan2_kernel_ts[grid](A_c, B, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        if out is None:
            out = torch.empty_like(
                B_c, dtype=_result_dtype(torch.tensor(A).dtype, B.dtype)
            )
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = B_c.numel()
        if n_elements == 0:
            if out_c.data_ptr() != out.data_ptr():
                out.copy_(out_c)
            return out

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        arctan2_kernel_st[grid](A, B_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    else:
        return torch.tensor(math.atan2(A, B))


def arctan2(input, other):
    return _arctan2_internal(input, other)


def arctan2_out(input, other, out):
    res = _arctan2_internal(input, other, out=out)
    return res


def arctan2_(input, other):
    # ATen rejects in-place arctan2 on integer and bool inputs (the float
    # result cannot be cast back into the output buffer); mirror that
    # instead of silently truncating.
    if not isinstance(input, torch.Tensor):
        raise TypeError("arctan2_: expected a Tensor as the first argument")
    if not input.dtype.is_floating_point:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type "
            f"{input.dtype}"
        )
    res = _arctan2_internal(input, other)
    input.copy_(res)
    return input
