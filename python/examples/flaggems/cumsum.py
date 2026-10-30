import math

import torch
import triton
import triton.language as tl

Tensor = torch.Tensor


@triton.jit
def program_id(axis: int) -> tl.tensor:
    return tl.program_id(axis).to(tl.int64)


# === 1D scan kernels（单列）===


@triton.jit
def scan_part_sum_kernel(
    inp,
    out,
    partial_sum,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp_vals = tl.load(inp + offset, mask=mask, other=0).to(
        partial_sum.dtype.element_ty
    )
    result = tl.cumsum(inp_vals, axis=0)
    part_sum = tl.sum(inp_vals)

    tl.store(out + offset, result, mask=mask)
    tl.store(partial_sum + pid, part_sum)


@triton.jit
def add_base_sum_kernel(
    out,
    partial_sum,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    out_vals = tl.load(out + offset, mask=mask, other=0)

    if pid > 0:
        last_part_sum = tl.load(partial_sum + pid - 1)

        final_vals = out_vals + last_part_sum
        tl.store(out + offset, final_vals.to(out_vals.dtype), mask=mask)


# === 2D/3D scan kernels（多列）===


@triton.jit
def scan_part_sum_abc_kernel(
    inp,
    out,
    partial_sum,
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
    inp_vals = tl.load(inp + offset, mask=mask, other=0).to(
        partial_sum.dtype.element_ty
    )
    result = tl.cumsum(inp_vals, axis=0)
    part_sum = tl.sum(inp_vals)

    tl.store(out + offset, result, mask=mask)
    tl.store(partial_sum + part_offset, part_sum)


@triton.jit
def add_base_sum_abc_kernel(
    out,
    partial_sum,
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
    out_vals = tl.load(out + offset, mask=mask, other=0)

    if pid_b > 0:
        last_part_sum = tl.load(partial_sum + last_part_offset)

        final_vals = out_vals + last_part_sum
        tl.store(out + offset, final_vals.to(out_vals.dtype), mask=mask)


# === reduce_then_scan_row（大 N 优化：单列）===


@triton.jit
def reduce_then_scan_block_sum_kernel_row(
    in_ptr,
    block_sum_ptr,
    N,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    pid_n = program_id(1)
    pid_m = program_id(0)
    num_programs_n = tl.num_programs(1)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)
    acc = tl.zeros((TILE_SIZE,), dtype=block_sum_ptr.dtype.element_ty)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(in_ptr + pid_m * N + offsets, mask=offsets < N, other=0).to(
            block_sum_ptr.dtype.element_ty
        )
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.store(block_sum_ptr + pid_m * num_programs_n + pid_n, block_sum)


@triton.jit
def reduce_then_scan_root_scan_kernel_row(in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr):
    pid = program_id(0)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=0).to(
        out_ptr.dtype.element_ty
    )
    out = tl.cumsum(x, 0)
    tl.store(out_ptr + pid * N + offsets, out, mask=mask)


@triton.jit
def reduce_then_scan_block_scan_kernel_row(
    in_ptr,
    previous_sum_ptr,
    out_ptr,
    N,
    num_tiles_n,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    pid_m = program_id(0)
    pid_n = program_id(1)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)

    prefix = tl.load(
        previous_sum_ptr + pid_m * num_tiles_n + pid_n - 1,
        mask=pid_n > 0,
        other=0,
    )
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + pid_m * N + offsets, mask=mask, other=0).to(
            previous_sum_ptr.dtype.element_ty
        )
        tile_scan = prefix + tl.cumsum(x, 0)
        prefix += tl.sum(x, 0)
        tl.store(out_ptr + pid_m * N + offsets, tile_scan, mask=mask)


def reduce_then_scan_row(x, out, M, N, compute_dtype):
    TILE_SIZE = triton.next_power_of_2(N)
    num_warps = 8 if TILE_SIZE > 2048 else 4

    if N <= 16384:
        reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
            x,
            out,
            N,
            TILE_SIZE,
            num_warps=num_warps,
        )
        return

    TILE_SIZE = min(4096, triton.next_power_of_2(N))
    num_warps = 8 if TILE_SIZE > 2048 else 4
    num_tiles = triton.cdiv(N, TILE_SIZE)
    # Simulate num_ctas on CPU: use fixed num_ctas based on num_tiles
    num_ctas = min(num_tiles, 4)
    ROOT_SCAN_TILE_SIZE = triton.next_power_of_2(num_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)

    block_sums = torch.empty((M, num_ctas), dtype=compute_dtype, device=x.device)
    block_inclusive_prefix = torch.empty_like(block_sums)

    reduce_then_scan_block_sum_kernel_row[(M, num_ctas, 1)](
        x,
        block_sums,
        N,
        tiles_per_cta,
        TILE_SIZE,
        num_warps=num_warps,
    )
    reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
        block_sums,
        block_inclusive_prefix,
        num_ctas,
        ROOT_SCAN_TILE_SIZE,
        num_warps=num_warps,
    )
    reduce_then_scan_block_scan_kernel_row[(M, num_ctas, 1)](
        x,
        block_inclusive_prefix,
        out,
        N,
        num_ctas,
        tiles_per_cta,
        TILE_SIZE,
        num_warps=num_warps,
    )


# === 分块扫描（col）===


def scan_then_fan_col(inp, out, n_ele, dtype):
    BLOCK_SIZE = triton.next_power_of_2(n_ele) if n_ele <= 1024 * 4 else 1024
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_sum = torch.empty(part_num, dtype=dtype, device=inp.device)

    grid = (part_num,)
    scan_part_sum_kernel[grid](
        inp,
        out,
        partial_sum,
        n_ele,
        part_num,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if part_num >= 2:
        scan_then_fan_col(partial_sum, partial_sum, part_num, dtype)
        add_base_sum_kernel[grid](
            out,
            partial_sum,
            n_ele,
            part_num,
            BLOCK_SIZE=BLOCK_SIZE,
        )


def scan_then_fan(inp, out, A, B, C, dtype):
    BLOCK_SIZE = triton.next_power_of_2(B) if B <= 1024 * 4 else 1024
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_sum = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)

    grid = (A, part_num, C)
    scan_part_sum_abc_kernel[grid](
        inp,
        out,
        partial_sum,
        B,
        C,
        part_num,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if part_num >= 2:
        scan_then_fan(partial_sum, partial_sum, A, part_num, C, dtype)
        add_base_sum_abc_kernel[grid](
            out,
            partial_sum,
            B,
            C,
            part_num,
            BLOCK_SIZE=BLOCK_SIZE,
        )


# === 公共接口 ===


def cumsum_wrapper(inp, dim=0, *, dtype=None, out=None):
    if inp.ndim == 0:
        if dim not in (-1, 0):
            raise IndexError("dimension out of range")
        if dtype is None:
            dtype = (
                out.dtype
                if out is not None
                else (
                    inp.dtype
                    if inp.is_floating_point() or inp.is_complex()
                    else torch.int64
                )
            )
        if out is not None and out.dtype != dtype:
            raise RuntimeError("cumsum: out dtype must match the requested dtype")
        result = inp.to(dtype).clone()
        if out is None:
            return result
        if out.shape != result.shape:
            out.resize_(result.shape)
        out.copy_(result)
        return out
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    inp = inp.contiguous()

    if dtype is None:
        dtype = (
            out.dtype
            if out is not None
            else (
                inp.dtype
                if inp.is_floating_point() or inp.is_complex()
                else torch.int64
            )
        )
    if out is not None and out.dtype != dtype:
        raise RuntimeError("cumsum: out dtype must match the requested dtype")
    # Match torch.cumsum's conversion before accumulation, including explicit
    # float-to-integer and float64-to-float32 dtype requests.
    inp = inp.to(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    elif dtype in (torch.int8, torch.int16, torch.uint8):
        # Use int32 consistently for narrow-integer block prefixes and
        # loop-carried accumulators.
        compute_dtype = torch.int32
    else:
        compute_dtype = dtype
    result = torch.empty(inp.shape, dtype=compute_dtype, device=inp.device)

    if inp.numel():
        shape = inp.shape
        M = math.prod(shape[:dim])
        N = shape[dim]
        K = inp.numel() // M // N
        if K == 1:
            reduce_then_scan_row(inp, result, M, N, compute_dtype)
        else:
            scan_then_fan(inp, result, M, N, K, compute_dtype)
    result = result.to(dtype)
    if out is None:
        return result
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out


def cumsum(inp, dim=0, *, dtype=None):
    return cumsum_wrapper(inp, dim, dtype=dtype)


def cumsum_out(inp, dim=0, *, dtype=None, out):
    return cumsum_wrapper(inp, dim, dtype=dtype, out=out)


def normed_cumsum(inp, dim=-1):
    assert inp.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = inp.numel() // N
    inp = inp.contiguous()
    out = torch.empty_like(inp)

    BLOCK = triton.next_power_of_2(N) if N <= 1024 * 4 else 1024

    grid = (M,)
    normed_cumsum_kernel[grid](inp, out, N, BLOCK=BLOCK)
    return out


@triton.jit
def normed_cumsum_kernel(inp, out, K, BLOCK: tl.constexpr):
    row_start = program_id(0) * K
    row_off = tl.arange(0, BLOCK)
    mask = row_off < K
    x = tl.load(inp + row_start + row_off, mask=mask, other=0.0)
    if tl.constexpr(x.dtype.is_fp16()):
        x = x.to(tl.float32)
    y_sum = tl.sum(x, 0)
    y = tl.cumsum(x, 0)
    y = y / y_sum
    tl.store(out + row_start + row_off, y, mask=mask)
