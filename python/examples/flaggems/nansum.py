import math

import torch
import triton
import triton.language as tl


@triton.jit
def nansum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    x = tl.load(inp_ptrs, mask=mask, other=0.0)
    x = tl.where(x != x, 0.0, x)
    # Accumulate in fp32 like the dim path; fp16 tl.sum loses precision.
    tl.store(mid + pid, tl.sum(x.to(tl.float32), axis=0))


@triton.jit
def nansum_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    tl.store(out, tl.sum(mid_val, axis=0))


@triton.jit
def nansum_kernel(
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
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(inp_ptrs + cols, mask=mask, other=0.0)
        acc += tl.where(a != a, 0.0, a)

    total = tl.sum(acc, axis=1)[:, None]
    tl.store(out_ptrs, total.to(out.dtype.element_ty), mask=row_mask)


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


def nansum(inp, dim=None, keepdim=False):
    if inp.numel() == 0:
        # torch returns zeros for empty reductions
        return torch.nansum(inp, dim=dim, keepdim=keepdim)
    if not inp.dtype.is_floating_point:
        return torch.nansum(inp, dim=dim, keepdim=keepdim)

    if inp.dtype == torch.float64:
        return torch.nansum(inp, dim=dim, keepdim=keepdim)

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=inp.dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        nansum_kernel_1[(mid_size,)](inp, mid, M, BLOCK_SIZE=block_size)
        nansum_kernel_2[(1,)](mid, out, mid_size, BLOCK_MID=block_mid)
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

        nansum_kernel[grid](inp, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

        if not keepdim:
            out = out.squeeze(dim=dim)
        return out
