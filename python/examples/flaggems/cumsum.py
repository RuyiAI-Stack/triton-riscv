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

    inp_vals = tl.load(inp + offset, mask=mask)
    if tl.constexpr(
        inp_vals.dtype.is_int64()
        or inp_vals.dtype.is_uint64()
        or inp_vals.dtype.is_fp64()
    ):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)

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

    out_vals = tl.load(out + offset, mask=mask)

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

    inp_vals = tl.load(inp + offset, mask=mask)
    if tl.constexpr(
        inp_vals.dtype.is_int64()
        or inp_vals.dtype.is_uint64()
        or inp_vals.dtype.is_fp64()
    ):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)

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

    out_vals = tl.load(out + offset, mask=mask)

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

    acc = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(in_ptr + pid_m * N + offsets, mask=offsets < N).to(
            tl.float32
        )
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.store(block_sum_ptr + pid_m * num_programs_n + pid_n, block_sum)


@triton.jit
def reduce_then_scan_root_scan_kernel_row(
    in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr
):
    pid = program_id(0)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=0).to(tl.float32)
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
    ).to(tl.float32)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + pid_m * N + offsets, mask=mask).to(tl.float32)
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

    block_sums = torch.empty(
        (M, num_ctas), dtype=compute_dtype, device=x.device
    )
    block_inclusive_prefix = torch.empty_like(block_sums)

    reduce_then_scan_block_sum_kernel_row[(M, num_ctas, 1, 1)](
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


@triton.jit
def cumsum_sequential_kernel(inp, out, B, C):
    pid_a = tl.program_id(0)
    pid_c = tl.program_id(1)
    base_offset = pid_a * B * C + pid_c

    if tl.constexpr(
        out.type.element_ty.is_fp16() or out.type.element_ty.is_bf16()
    ):
        compute_dtype = tl.float32
    else:
        compute_dtype = out.type.element_ty

    total = tl.full((), 0, dtype=compute_dtype)
    for b_idx in range(0, B):
        offset = base_offset + b_idx * C
        total += tl.load(inp + offset).to(compute_dtype)
        tl.store(out + offset, total.to(out.type.element_ty))


def cumsum_wrapper(inp, dim=0, *, dtype=None, out=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    inp = inp.contiguous()

    if dtype is None:
        dtype = torch.int64 if not inp.dtype.is_floating_point else inp.dtype
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    if inp.numel() == 0:
        return out

    shape = inp.shape
    M = math.prod(shape[:dim])
    N = shape[dim]
    K = inp.numel() // M // N

    cumsum_sequential_kernel[(M, K)](inp, out, N, K)

    return out


def cumsum(inp, dim=0, *, dtype=None):
    return cumsum_wrapper(inp, dim, dtype=dtype)


def cumsum_out(inp, dim=0, *, dtype=None, out):
    return cumsum_wrapper(inp, dim, dtype=dtype, out=out)


def normed_cumsum(inp, dim=-1):
    assert inp.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    shape = inp.shape
    dim = dim % inp.ndim
    A = math.prod(shape[:dim])
    B = shape[dim]
    C = math.prod(shape[dim + 1 :])
    inp = inp.contiguous()

    out = torch.empty_like(inp)
    normed_cumsum_kernel[(A, C)](inp, out, B, C)
    return out


@triton.jit
def normed_cumsum_kernel(inp, out, B, C):
    pid_a = tl.program_id(0)
    pid_c = tl.program_id(1)
    base_offset = pid_a * B * C + pid_c
    if tl.constexpr(
        inp.type.element_ty.is_fp16() or inp.type.element_ty.is_bf16()
    ):
        compute_dtype = tl.float32
    else:
        compute_dtype = inp.type.element_ty

    row_sum = tl.full((), 0, dtype=compute_dtype)
    for b_idx in range(0, B):
        offset = base_offset + b_idx * C
        row_sum += tl.load(inp + offset).to(compute_dtype)

    prefix = tl.full((), 0, dtype=compute_dtype)
    for b_idx in range(0, B):
        offset = base_offset + b_idx * C
        prefix += tl.load(inp + offset).to(compute_dtype)
        tl.store(out + offset, (prefix / row_sum).to(out.type.element_ty))
