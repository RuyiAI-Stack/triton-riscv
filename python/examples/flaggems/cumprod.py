import math

import torch
import triton
import triton.language as tl


Tensor = torch.Tensor


@triton.jit
def reduce_mul(a, b):
    return a * b


@triton.jit
def program_id(axis: int) -> tl.tensor:
    return tl.program_id(axis).to(tl.int64)


@triton.jit
def scan_part_product_kernel(
    inp,
    out,
    partial_product,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask, other=1)
    if tl.constexpr(
        inp_vals.dtype.is_int64()
        or inp_vals.dtype.is_uint64()
        or inp_vals.dtype.is_fp64()
    ):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int64)
    else:
        inp_vals = inp_vals.to(tl.float32)

    result = tl.cumprod(inp_vals, axis=0)
    part_prod = tl.reduce(inp_vals, axis=0, combine_fn=reduce_mul)

    tl.store(out + offset, result, mask=mask)
    tl.store(partial_product + pid, part_prod)


@triton.jit
def multiply_base_product_kernel(
    out,
    partial_product,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_vals = tl.load(out + offset, mask=mask)

    if pid > 0:
        base_product = tl.load(partial_product + pid - 1)
        final_vals = out_vals * base_product
        tl.store(out + offset, final_vals, mask=mask)


def scan_then_fan_col(inp, out, n_ele, dtype):
    BLOCK_SIZE = triton.next_power_of_2(n_ele) if n_ele <= 1024 * 4 else 1024
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_product = torch.empty(part_num, dtype=dtype, device=inp.device)

    grid = (part_num,)
    scan_part_product_kernel[grid](
        inp,
        out,
        partial_product,
        n_ele,
        part_num,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if part_num >= 2:
        partial_prefix = torch.empty_like(partial_product)
        scan_then_fan_col(partial_product, partial_prefix, part_num, dtype)
        multiply_base_product_kernel[grid](
            out,
            partial_prefix,
            n_ele,
            part_num,
            BLOCK_SIZE=BLOCK_SIZE,
        )


@triton.jit
def scan_part_product_abc_kernel(
    inp,
    out,
    partial_product,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = program_id(0)
    pid_b = program_id(1)
    pid_c = program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    part_offset = base_part_offset + pid_b * C
    mask = b_idx < B

    inp_vals = tl.load(inp + offset, mask=mask, other=1)
    if tl.constexpr(
        inp_vals.dtype.is_int64()
        or inp_vals.dtype.is_uint64()
        or inp_vals.dtype.is_fp64()
    ):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int64)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl.cumprod(inp_vals, axis=0)
    part_prod = tl.reduce(inp_vals, axis=0, combine_fn=reduce_mul)

    tl.store(out + offset, result, mask=mask)
    tl.store(partial_product + part_offset, part_prod)


@triton.jit
def multiply_base_product_abc_kernel(
    out,
    partial_product,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = program_id(0)
    pid_b = program_id(1)
    pid_c = program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C
    mask = b_idx < B

    out_vals = tl.load(out + offset, mask=mask)

    if pid_b > 0:
        base_product = tl.load(partial_product + last_part_offset)
        final_vals = out_vals * base_product
        tl.store(out + offset, final_vals, mask=mask)


def scan_then_fan(inp, out, A, B, C, dtype):
    BLOCK_SIZE = triton.next_power_of_2(B) if B <= 1024 * 4 else 1024
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_product = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)

    grid = (A, part_num, C)
    scan_part_product_abc_kernel[grid](
        inp,
        out,
        partial_product,
        B,
        C,
        part_num,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if part_num >= 2:
        partial_prefix = torch.empty_like(partial_product)
        scan_then_fan(partial_product, partial_prefix, A, part_num, C, dtype)
        multiply_base_product_abc_kernel[grid](
            out,
            partial_prefix,
            B,
            C,
            part_num,
            BLOCK_SIZE=BLOCK_SIZE,
        )


def cumprod(inp, dim: int):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    inp = inp.contiguous()

    shape = inp.shape
    M = math.prod(shape[:dim])
    N = shape[dim]
    K = inp.numel() // M // N

    dtype = torch.int64 if not inp.dtype.is_floating_point else inp.dtype
    out = torch.empty_like(inp, dtype=dtype)

    compute_dtype = (
        torch.float32 if inp.dtype in (torch.float16, torch.bfloat16) else dtype
    )
    if not inp.dtype.is_floating_point:
        compute_dtype = torch.int64

    if inp.numel() == 0:
        return out

    if K == 1:
        scan_then_fan(inp, out, M, N, 1, compute_dtype)
    else:
        scan_then_fan(inp, out, M, N, K, compute_dtype)

    return out


def cumprod_(inp, dim: int):
    res = cumprod(inp, dim)
    inp.copy_(res)
    return inp
