import math

import torch
import triton
import triton.language as tl

from .unique import _unique2


@triton.jit
def reduce_all(a, b):
    return a and b


@triton.jit
def reduce_any(a, b):
    return a or b


def launch_arg(BLOCK_M, BLOCK_N, N, num_warps):
    return BLOCK_M, min(BLOCK_N, triton.next_power_of_2(N)), num_warps


@triton.jit
def isin_by_comparation_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    invert: tl.constexpr,
):
    row_off = global_pid * BLOCK_M
    rows = row_off + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    out_ptr += rows
    in0_ravel_ptr += rows + tl.zeros([BLOCK_N], dtype=tl.int32)
    in1_ravel_ptr += tl.zeros([BLOCK_M], dtype=tl.int32)[:, None]

    block = tl.full(
        [BLOCK_M, BLOCK_N], value=(1 if invert else 0), dtype=tl.int1
    )
    in0 = tl.load(in0_ravel_ptr, row_mask, other=0)
    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        in1 = tl.load(in1_ravel_ptr + cols, mask, other=0)
        block = tl.where(
            mask,
            tl.where(invert, block and (in0 != in1), block or (in0 == in1)),
            invert,
        )
    out = tl.reduce(
        block, axis=1, combine_fn=(reduce_all if invert else reduce_any)
    )
    tl.store(out_ptr, out[:, None], row_mask)


@triton.jit
def isin_by_comparation_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    tiles_per_cta: int,
    invert: tl.constexpr,
):
    pid = tl.program_id(0)
    ctas_num = tl.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        isin_by_comparation_impl(
            global_pid,
            in0_ravel_ptr,
            in1_ravel_ptr,
            out_ptr,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            invert,
        )


@triton.jit
def isin_scalar_kernel(in0_ptr, in1_ptr, out_ptr, N, invert: tl.constexpr):
    element_idx = tl.program_id(0)
    value = tl.load(in0_ptr + element_idx)
    found = tl.full((), False, dtype=tl.int1)
    for test_idx in range(0, N):
        found |= value == tl.load(in1_ptr + test_idx)
    result = ~found if invert else found
    tl.store(out_ptr + element_idx, result.to(tl.int8))


def isin_by_comparation(
    in0: torch.tensor,
    in1: torch.tensor,
    invert: bool,
):
    in0_ravel = in0.contiguous().ravel()
    in1_ravel = in1.contiguous().ravel()
    M = in0.numel()
    N = in1.numel()
    out = torch.empty_like(in0_ravel, dtype=torch.uint8)
    isin_scalar_kernel[(M,)](in0_ravel, in1_ravel, out, N, invert=invert)
    return out.bool().view_as(in0)


@triton.jit
def isin_by_search_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_sorted_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    log_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    invert: tl.constexpr,
):
    r = tl.arange(0, BLOCK_M)
    i0 = global_pid * BLOCK_M + r
    mask = i0 < M

    in0_ravel = tl.load(in0_ravel_ptr + i0, mask=mask)

    out = tl.zeros_like(r).to(tl.int1)
    start = tl.zeros_like(r)
    end = start + N
    while_mask = start < end
    for i in range(log_n):
        mid = tl.where(while_mask, start + (end - start) // 2, 0)
        mid_val = tl.load(in1_sorted_ptr + mid, mask=while_mask)
        out = tl.where(while_mask, out or (mid_val == in0_ravel), out)
        start = tl.where(while_mask and (mid_val < in0_ravel), mid + 1, start)
        end = tl.where(while_mask and (mid_val > in0_ravel), mid, end)
        while_mask = start < end

    tl.store(out_ptr + i0, not out if invert else out, mask=mask)


@triton.jit
def isin_by_search_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_sorted_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    log_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    tiles_per_cta: int,
    invert: tl.constexpr,
):
    pid = tl.program_id(0)
    ctas_num = tl.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        isin_by_search_impl(
            global_pid,
            in0_ravel_ptr,
            in1_sorted_ptr,
            out_ptr,
            M,
            N,
            log_n,
            BLOCK_M,
            invert,
        )


def isin_by_search(
    in0: torch.tensor,
    in1: torch.tensor,
    invert: bool,
    unique_in0: bool,
    unique_in1: bool,
):
    if unique_in0:
        in0_ravel, unique_order, _ = _unique2(
            in0, sorted=True, return_inverse=True, return_counts=False
        )
    else:
        in0_ravel = in0.contiguous().ravel()
    if unique_in1:
        in1_ravel, _, _ = _unique2(
            in1, sorted=True, return_inverse=False, return_counts=False
        )
    else:
        in1_ravel, _ = torch.sort(in1.ravel())
    M = in0_ravel.numel()
    N = in1_ravel.numel()
    if M <= 1048576:
        _, BLOCK_M, num_warps = launch_arg(None, 512, M, 8)
    elif M <= 4194304:
        _, BLOCK_M, num_warps = launch_arg(None, 1024, M, 8)
    elif M <= 8388608:
        _, BLOCK_M, num_warps = launch_arg(None, 2048, M, 16)
    elif M <= 268435456:
        _, BLOCK_M, num_warps = launch_arg(None, 4096, M, 32)
    else:
        _, BLOCK_M, num_warps = launch_arg(None, 2048, M, 16)
    log_n = int(math.log2(N)) + 1
    ctas_num = min(65536, triton.cdiv(M, BLOCK_M))
    tiles_per_cta = triton.cdiv(M, BLOCK_M * ctas_num)
    grid = (ctas_num,)
    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    isin_by_search_kernel[grid](
        in0_ravel,
        in1_ravel,
        out,
        M,
        N,
        log_n,
        BLOCK_M,
        tiles_per_cta=tiles_per_cta,
        invert=invert,
        num_warps=num_warps,
    )
    if unique_in0:
        out = torch.gather(out, 0, unique_order.ravel().to(torch.int64))
    return out.view_as(in0)


def isin(
    in0,
    in1,
    *,
    assume_unique: bool = False,
    invert: bool = False,
):
    if not torch.is_tensor(in0):
        assert torch.is_tensor(in1)
        in0 = torch.tensor(in0, device=in1.device)
    elif not torch.is_tensor(in1):
        assert torch.is_tensor(in0)
        in1 = torch.tensor(in1, device=in0.device)
    if in0.numel() == 0 or in1.numel() == 0:
        return torch.full_like(in0, invert, dtype=torch.bool)
    elif in0.numel() <= 12288 and in1.numel() <= 12288:
        return isin_by_comparation(in0, in1, invert)
    elif assume_unique or in1.numel() <= 4194304:
        return isin_by_search(
            in0, in1, invert, unique_in0=False, unique_in1=False
        )
    else:
        return isin_by_search(
            in0, in1, invert, unique_in0=False, unique_in1=True
        )
