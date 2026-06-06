import math

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mul(a, b):
    return a * b


@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))


@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul)
    tl.store(out, prod_val)


def prod(inp, *, dtype=None):
    if dtype is None:
        dtype = inp.dtype

    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    prod_kernel_mid[(mid_size, 1, 1)](inp, mid, M, block_size)
    prod_kernel_result[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


@triton.jit
def prod_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.full((BLOCK_M, BLOCK_N), value=1.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]

        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
        acc *= inp_vals
    result_index = tl.reduce(acc, axis=1, combine_fn=reduce_mul)

    offset_index = m_offset
    out_ptrs = out + offset_index
    mask1 = m_offset < M
    tl.store(out_ptrs, result_index, mask=mask1)


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    N = shape[dim]
    inp = dim_compress(inp, dim)
    N = inp.shape[1]
    shape[dim] = 1
    M = inp.numel() // N

    if dtype is None:
        dtype = inp.dtype
    out = torch.empty(shape, dtype=dtype, device=inp.device)
    if not keepdim:
        out = torch.squeeze(out, dim)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    prod_dim_kernel[grid](inp, out, M, N, BLOCK_M=1, BLOCK_N=triton.next_power_of_2(N))
    return out
