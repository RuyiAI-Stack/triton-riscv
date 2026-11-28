import math

import torch
import triton
import triton.language as tl


@triton.jit
def nanmean_kernel_1(
    inp,
    mid_sum,
    mid_cnt,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    x = tl.load(inp + offset, mask=mask, other=0.0)
    # Padding lanes load 0.0 and must not count as valid elements.
    valid = (x == x) & mask
    x = tl.where(valid, x, 0.0)
    # Accumulate in fp32; fp16 tl.sum loses precision on long blocks.
    tl.store(mid_sum + pid, tl.sum(x.to(tl.float32), axis=0))
    tl.store(mid_cnt + pid, tl.sum(valid.to(tl.float32), axis=0))


@triton.jit
def nanmean_kernel_2(
    mid_sum,
    mid_cnt,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    s = tl.sum(tl.load(mid_sum + offset, mask=mask, other=0.0), axis=0)
    c = tl.sum(tl.load(mid_cnt + offset, mask=mask, other=0.0), axis=0)
    # All-NaN input has zero valid elements and must produce NaN.
    mean = tl.where(c == 0.0, float("nan"), s / c)
    tl.store(out, mean.to(out.dtype.element_ty))


@triton.jit
def nanmean_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp_ptrs = inp + rows * N
    out_ptrs = out + rows
    row_mask = rows < M

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    cnt = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(inp_ptrs + cols, mask=mask, other=0.0)
        # Padding lanes must not count as valid elements.
        valid = (a == a) & mask
        acc += tl.where(valid, a, 0.0)
        cnt += valid.to(tl.float32)

    s = tl.sum(acc, axis=1)[:, None]
    c = tl.sum(cnt, axis=1)[:, None]
    mean = tl.where(c == 0.0, float("nan"), s / c)
    tl.store(out_ptrs, mean.to(out.dtype.element_ty), mask=row_mask)


def _dim_compress(inp, dims):
    if not isinstance(dims, (list, tuple)):
        dims = [dims]

    dims = [d % inp.ndim for d in dims]
    other_dims = [i for i in range(inp.ndim) if i not in dims]
    perm = other_dims + dims

    inp_perm = inp.permute(perm).contiguous()

    N = 1
    for d in dims:
        N *= inp.shape[d]
    M = inp.numel() // N

    return inp_perm.view(M, N)


def nanmean(inp, dim=None, keepdim=False):
    if inp.numel() == 0:
        # torch returns NaN for empty reductions
        return torch.nanmean(inp, dim=dim, keepdim=keepdim)
    if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.nanmean(inp, dim=dim, keepdim=keepdim)

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid_sum = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
        mid_cnt = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=inp.dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        nanmean_kernel_1[(mid_size,)](inp, mid_sum, mid_cnt, M, BLOCK_SIZE=block_size)
        nanmean_kernel_2[(1,)](mid_sum, mid_cnt, out, mid_size, BLOCK_MID=block_mid)
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        for i in dim:
            assert i >= -inp.ndim and i < inp.ndim, "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = _dim_compress(inp, dim)

        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M),)

        nanmean_kernel[grid](inp, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

        if not keepdim:
            out = out.squeeze(dim=dim)
        return out
