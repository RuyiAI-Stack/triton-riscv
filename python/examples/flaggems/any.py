import math

import torch
import triton
import triton.language as tl


# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok.


@triton.jit
def reduce_any(a, b):
    return a or b


@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp_ptrs = inp + rows * N
    out_ptrs = out + rows
    row_mask = rows < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(inp_ptrs + cols, mask=mask, other=0.0)
        _any = _any | (a != 0)
    any_res = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out_ptrs, any_res[:, None].to(tl.int8), mask=row_mask)


@triton.jit
def any_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    any_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_any)
    mid_ptr = mid + pid
    tl.store(mid_ptr, any_val.to(tl.int8))


@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.int1)
    any_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val.to(tl.int8))


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def any(inp):
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.uint8, device=inp.device)
    out = torch.empty([], dtype=torch.uint8, device=inp.device)

    any_kernel_1[(mid_size,)](inp, mid, n_elements, BLOCK_SIZE=block_size)
    any_kernel_2[(1,)](mid, out, mid_size, BLOCK_MID=block_mid)

    return out.to(torch.bool)


def any_dim(inp, dim=None, keepdim=False):
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp_compressed = dim_compress(inp, [dim])
        N = shape[dim]
        shape[dim] = 1
        M = inp_compressed.numel() // N

        out = torch.empty(shape, dtype=torch.uint8, device=inp.device)

        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        any_kernel_dim[grid](
            inp_compressed,
            out,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        out = out.to(torch.bool)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    for i in dim:
        assert i >= -inp.ndim and i < inp.ndim, "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp_compressed = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp_compressed.numel() // N

    out = torch.empty(shape, dtype=torch.uint8, device=inp.device)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M),)
    any_kernel_dim[grid](
        inp_compressed,
        out,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    out = out.to(torch.bool)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
