import math
from functools import reduce

import torch
import triton
import triton.language as tl


@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@triton.jit
def mean_kernel_2(
    mid,
    out,
    M,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    mean_val = sum_val / M
    tl.store(out, mean_val)


@triton.jit
def mean_dim_kernel_non_inner_vec(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    input_dtype = input_ptr.dtype.element_ty
    if tl.constexpr(input_dtype == tl.float16) or tl.constexpr(
        input_dtype == tl.bfloat16
    ):
        ACC_DTYPE = tl.float32
    else:
        ACC_DTYPE = input_dtype

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_base = pid_k * BLOCK_SIZE_K * VEC_SIZE
    k_offsets = (
        k_base
        + tl.arange(0, BLOCK_SIZE_K)[:, None] * VEC_SIZE
        + tl.arange(0, VEC_SIZE)[None, :]
    )
    k_mask = k_offsets < K

    acc = tl.zeros((BLOCK_SIZE_K, VEC_SIZE), dtype=ACC_DTYPE)

    base = pid_m * N * K

    for n in range(N):
        offsets = base + n * K + k_offsets
        val = tl.load(input_ptr + offsets, mask=k_mask, other=0.0)
        acc += val.to(ACC_DTYPE)

    mean_val = acc / N

    out_offsets = pid_m * K + k_offsets
    tl.store(output_ptr + out_offsets, mean_val, mask=k_mask)


@triton.jit
def mean_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        summed = tl.sum(inp, axis=0, keep_dims=True)
        out = summed / N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        sum_tile = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(
                cdtype
            )
            sum_tile += inp
        summed = tl.sum(sum_tile, axis=0, keep_dims=True)
        out = summed / N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


@triton.jit
def mean_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + inp_offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        summed = tl.sum(inp, axis=0)
        out = summed / N
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)
    else:
        sum_vec = tl.zeros([TILE_N], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(
                cdtype
            )
            sum_vec += inp
        summed = tl.sum(sum_vec, axis=0)
        out = summed / N
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)


@triton.jit
def mean_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp_ptrs = inp + rows * N
    out_ptrs = out + rows
    row_mask = rows < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(inp_ptrs + cols, mask, other=0).to(cdtype)
        _sum += a
    summed = tl.sum(_sum, axis=1)[:, None]
    mean = summed / N
    tl.store(out_ptrs, mean, row_mask)


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def mean(inp, *, dtype=None):
    inp = inp.contiguous()
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
    mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


def mean_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64

    if dim == []:
        if not keepdim:
            return mean(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(mean(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)

    if isinstance(dim, int):
        dim = [dim]
    else:
        try:
            dim = list(dim)
        except TypeError:
            raise TypeError(
                f"dim must be an int, iterable of ints, or [], got {type(dim)}"
            )

    dim = [d % inp.ndim for d in dim]

    if len(dim) == 1:
        dim0 = dim[0]
        N = inp.shape[dim0]
        M = reduce(lambda x, y: x * y, shape[:dim0], 1)
        inp = inp.contiguous()
        K = inp.numel() // M // N
        shape[dim0] = 1
        if out is None:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        if K >= 1024:
            input_dtype = inp.dtype
            if input_dtype in (torch.float16, torch.bfloat16):
                VEC_SIZE = 8
                BLOCK_SIZE_K = 128
            else:
                VEC_SIZE = 1
                BLOCK_SIZE_K = min(triton.next_power_of_2(K), 512)
            grid = (M, triton.cdiv(K, BLOCK_SIZE_K * VEC_SIZE))
            mean_dim_kernel_non_inner_vec[grid](
                out,
                inp,
                M,
                N,
                K,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                VEC_SIZE=VEC_SIZE,
                num_warps=8 if BLOCK_SIZE_K <= 128 else 16,
            )
        elif K > 1:
            TILE_N = 32
            TILE_K = min(triton.next_power_of_2(K), 256)
            grid = (M, triton.cdiv(K, TILE_K), 1)
            mean_dim_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                ONE_TILE_PER_CTA=K <= TILE_N,
            )
        else:
            TILE_N = triton.next_power_of_2(N)
            grid = (M, 1, 1)
            mean_dim_kernel_inner[grid](
                out,
                inp,
                M,
                N,
                TILE_N=TILE_N,
                ONE_TILE_PER_CTA=N <= TILE_N,
            )
        if not keepdim:
            out = out.squeeze(dim=dim0)
        return out
    else:
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N
        if out is None:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        mean_dim_kernel[grid](inp, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def mean_dim(inp, dim=None, keepdim=False, *, dtype=None):
    return mean_dim_comm(inp, dim, keepdim, dtype=dtype)
