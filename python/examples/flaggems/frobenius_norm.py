import math

import torch
import triton
import triton.language as tl


@triton.jit
def frobenius_norm_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    x_ptrs = X + rows * N
    out_ptrs = Out + rows
    row_mask = rows < M

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(x_ptrs + cols, mask=mask, other=0.0)
        acc += a * a

    norm = tl.sqrt(tl.sum(acc, axis=1)[:, None])
    tl.store(out_ptrs, norm.to(Out.dtype.element_ty), mask=row_mask)


@triton.jit
def frobenius_norm_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    x = tl.load(inp + offset, mask=mask, other=0.0)
    tl.store(mid + pid, tl.sum(x * x, axis=0))


@triton.jit
def frobenius_norm_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    v = tl.load(mid + offset, mask=mask, other=0.0)
    tl.store(out, tl.sqrt(tl.sum(v, axis=0)))


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


def frobenius_norm(x, dim=None, keepdim=False):
    if x.numel() == 0:
        # frobenius over a 1-tuple dim is the L2 vector norm
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)
    if x.dtype != torch.float32:
        # frobenius over a 1-tuple dim is the L2 vector norm
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)

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
        frobenius_norm_kernel_1[(mid_size,)](x, mid, M, BLOCK_SIZE=block_size)
        frobenius_norm_kernel_2[(1,)](mid, out, mid_size, BLOCK_MID=block_mid)
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        shape = list(x.shape)
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
        frobenius_norm_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

        if not keepdim:
            out = out.squeeze(dim=dim)
        return out
