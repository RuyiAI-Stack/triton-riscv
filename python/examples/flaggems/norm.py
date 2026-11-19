import math

import torch
import triton
import triton.language as tl


@triton.jit
def norm_stage1(
    inp,
    mid,
    ord,
    M,
    CODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # CODE: 2 = l2, 1 = +inf, -1 = -inf, 0 = l0, 100 = generic |x|^ord.
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    if CODE == -1:
        # Neutral element for a min-reduction over |x| is +inf.
        x = tl.load(inp + offset, mask=mask, other=float("inf"))
        acc = tl.min(tl.abs(x))
    else:
        x = tl.load(inp + offset, mask=mask, other=0.0)
        if CODE == 2:
            x32 = x.to(tl.float32)
            acc = tl.sum(x32 * x32)
        elif CODE == 1:
            acc = tl.max(tl.abs(x))
        elif CODE == 0:
            acc = tl.sum(((x != 0) & mask).to(tl.float32))
        else:
            # Gate padding lanes: |0| -> log -> -inf, and a negative ord
            # turns exp(negative * -inf) into inf, poisoning the sum.
            ax = tl.abs(x)
            acc = tl.sum(tl.where(mask, tl.exp(ord * tl.log(ax)), 0.0))
    tl.store(mid + pid, acc)


@triton.jit
def norm_stage2(
    mid,
    out,
    ord,
    MID_SIZE,
    CODE: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    if CODE == -1:
        v = tl.load(mid + offset, mask=mask, other=float("inf"))
        tl.store(out, tl.min(v))
    else:
        v = tl.load(mid + offset, mask=mask, other=0.0)
        if CODE == 2:
            tl.store(out, tl.sqrt(tl.sum(v)))
        elif CODE == 1:
            tl.store(out, tl.max(v))
        elif CODE == 0:
            tl.store(out, tl.sum(v))
        else:
            s = tl.sum(v)
            tl.store(out, tl.exp(tl.log(s) / ord))


@triton.jit
def norm_row_kernel(
    inp,
    out,
    M,
    N,
    ord,
    CODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp_ptrs = inp + rows * N
    out_ptrs = out + rows
    row_mask = rows < M

    if CODE == -1:
        acc = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        if CODE == -1:
            # Padding must stay at the +inf neutral element.
            a = tl.load(inp_ptrs + cols, mask=mask, other=float("inf"))
        else:
            a = tl.load(inp_ptrs + cols, mask=mask, other=0.0)
        if CODE == 2:
            a32 = a.to(tl.float32)
            acc += a32 * a32
        elif CODE == 1:
            acc = tl.maximum(acc, tl.abs(a))
        elif CODE == -1:
            acc = tl.minimum(acc, tl.abs(a))
        elif CODE == 0:
            acc += (a != 0) & mask
            acc = acc.to(tl.float32)
        else:
            # Same padding gate as the global kernel for negative ord.
            ax = tl.abs(a)
            acc += tl.where(mask, tl.exp(ord * tl.log(ax)), 0.0)

    if CODE == 2:
        res = tl.sqrt(tl.sum(acc, axis=1)[:, None])
    elif CODE == 1:
        res = tl.max(acc, axis=1)[:, None]
    elif CODE == -1:
        res = tl.min(acc, axis=1)[:, None]
    elif CODE == 0:
        res = tl.sum(acc, axis=1)[:, None]
    else:
        s = tl.sum(acc, axis=1)[:, None]
        res = tl.exp(tl.log(s) / ord)
    tl.store(out_ptrs, res.to(out.dtype.element_ty), mask=row_mask)


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


def _ord_code(p):
    if p == 2:
        return 2
    if p == float("inf"):
        return 1
    if p == float("-inf"):
        return -1
    if p == 0:
        return 0
    return 100


def norm(x, p=2, dim=None, keepdim=False):
    if not x.dtype.is_floating_point:
        # torch: linalg.vector_norm expects a floating point tensor
        raise NotImplementedError(
            "linalg.vector_norm: Expected a floating point or complex tensor"
        )
    if x.numel() == 0:
        return torch.linalg.norm(x, ord=p, dim=dim, keepdim=keepdim)
    if x.dtype == torch.float64:
        return torch.linalg.norm(x, ord=p, dim=dim, keepdim=keepdim)

    code = _ord_code(p)
    if dim is None:
        M = x.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
        if not keepdim:
            out = torch.empty([], dtype=x.dtype, device=x.device)
        else:
            shape = [1] * x.dim()
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
        norm_stage1[(mid_size,)](x, mid, p, M, CODE=code, BLOCK_SIZE=block_size)
        norm_stage2[(1,)](mid, out, p, mid_size, CODE=code, BLOCK_MID=block_mid)
        return out
    else:
        shape = list(x.shape)
        if isinstance(dim, int):
            dim = [dim]
        dim = [d % x.ndim for d in dim]
        x = _dim_compress(x, dim)

        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N

        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        norm_row_kernel[grid](
            x, out, M, N, p, CODE=code, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out
